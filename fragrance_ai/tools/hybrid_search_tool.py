"""
하이브리드 검색 도구 - PostgreSQL pgvector 기반
벡터 유사도 검색 + 전통적 필터링 조합
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sqlalchemy import and_, or_, func, text
from sqlalchemy.orm import Session
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass

from fragrance_ai.database.schema import (
    Fragrance, Note, FragranceComposition,
    KnowledgeBase, DNASequence
)
from fragrance_ai.database.connection import DatabaseManager


@dataclass
class SearchResult:
    """검색 결과"""
    id: int
    type: str  # fragrance, note, knowledge
    name: str
    description: str
    similarity: float
    metadata: Dict[str, Any]


class HybridSearchTool:
    """하이브리드 검색 도구 - 실제 벡터 검색"""

    def __init__(self, db_manager: DatabaseManager = None):
        """
        초기화

        Args:
            db_manager: 데이터베이스 매니저
        """
        self.db_manager = db_manager or DatabaseManager()

        # 임베딩 모델 로드
        print("하이브리드 검색 임베딩 모델 로드 중...")
        self.encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        # pgvector 확인
        if not self.db_manager.check_pgvector():
            print("경고: pgvector가 설치되지 않음. 벡터 검색이 제한됩니다.")

    def search(
        self,
        query: str,
        search_type: str = "all",  # all, fragrance, note, knowledge
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        similarity_threshold: float = 0.3
    ) -> List[SearchResult]:
        """
        하이브리드 검색 실행

        Args:
            query: 검색 쿼리
            search_type: 검색 대상 타입
            filters: 추가 필터 조건
            top_k: 반환할 결과 수
            similarity_threshold: 유사도 임계값

        Returns:
            검색 결과 리스트
        """
        # 쿼리 임베딩
        query_embedding = self.encoder.encode(query)

        results = []

        with self.db_manager.get_session() as session:
            # 검색 타입에 따라 다른 테이블 검색
            if search_type in ["all", "fragrance"]:
                fragrance_results = self._search_fragrances(
                    session, query_embedding, filters, top_k, similarity_threshold
                )
                results.extend(fragrance_results)

            if search_type in ["all", "note"]:
                note_results = self._search_notes(
                    session, query_embedding, filters, top_k, similarity_threshold
                )
                results.extend(note_results)

            if search_type in ["all", "knowledge"]:
                knowledge_results = self._search_knowledge(
                    session, query_embedding, filters, top_k, similarity_threshold
                )
                results.extend(knowledge_results)

        # 유사도로 정렬
        results.sort(key=lambda x: x.similarity, reverse=True)

        return results[:top_k]

    def _search_fragrances(
        self,
        session: Session,
        query_embedding: np.ndarray,
        filters: Optional[Dict[str, Any]],
        top_k: int,
        threshold: float
    ) -> List[SearchResult]:
        """향수 검색"""
        results = []

        try:
            # pgvector를 사용한 코사인 유사도 검색
            # <=> 연산자는 L2 거리, <#> 는 내적, <-> 는 코사인 거리
            query = session.query(
                Fragrance,
                (1 - Fragrance.embedding.cosine_distance(query_embedding)).label("similarity")
            )

            # 필터 적용
            if filters:
                if "family" in filters:
                    query = query.filter(Fragrance.family == filters["family"])
                if "gender" in filters:
                    query = query.filter(Fragrance.gender == filters["gender"])
                if "brand" in filters:
                    query = query.filter(Fragrance.brand == filters["brand"])

            # 유사도 필터링 및 정렬
            query = query.filter(
                text(f"1 - (embedding <-> :query_embedding) > {threshold}")
            ).params(
                query_embedding=f"[{','.join(map(str, query_embedding))}]"
            ).order_by(
                text("similarity DESC")
            ).limit(top_k)

            for fragrance, similarity in query:
                # 조합 정보 가져오기
                compositions = session.query(
                    Note.name,
                    FragranceComposition.percentage,
                    FragranceComposition.pyramid_level
                ).join(
                    FragranceComposition
                ).filter(
                    FragranceComposition.fragrance_id == fragrance.id
                ).all()

                metadata = {
                    "family": fragrance.family,
                    "gender": fragrance.gender,
                    "brand": fragrance.brand,
                    "year": fragrance.year,
                    "compositions": [
                        {
                            "note": comp[0],
                            "percentage": comp[1],
                            "level": comp[2]
                        }
                        for comp in compositions
                    ]
                }

                results.append(SearchResult(
                    id=fragrance.id,
                    type="fragrance",
                    name=fragrance.name,
                    description=fragrance.description or "",
                    similarity=float(similarity),
                    metadata=metadata
                ))

        except Exception as e:
            print(f"향수 검색 오류: {e}")
            # 폴백: 전통적 검색
            results = self._fallback_fragrance_search(
                session, query_embedding, filters, top_k
            )

        return results

    def _search_notes(
        self,
        session: Session,
        query_embedding: np.ndarray,
        filters: Optional[Dict[str, Any]],
        top_k: int,
        threshold: float
    ) -> List[SearchResult]:
        """향료 노트 검색"""
        results = []

        try:
            query = session.query(
                Note,
                (1 - Note.embedding.cosine_distance(query_embedding)).label("similarity")
            )

            # 필터 적용
            if filters:
                if "type" in filters:
                    query = query.filter(Note.type == filters["type"])
                if "pyramid_level" in filters:
                    query = query.filter(Note.pyramid_level == filters["pyramid_level"])
                if "is_natural" in filters:
                    query = query.filter(Note.is_natural == filters["is_natural"])

            # 유사도 필터링 및 정렬
            query = query.filter(
                text(f"1 - (embedding <-> :query_embedding) > {threshold}")
            ).params(
                query_embedding=f"[{','.join(map(str, query_embedding))}]"
            ).order_by(
                text("similarity DESC")
            ).limit(top_k)

            for note, similarity in query:
                metadata = {
                    "type": note.type,
                    "pyramid_level": note.pyramid_level,
                    "volatility": note.volatility,
                    "strength": note.strength,
                    "longevity": note.longevity,
                    "is_natural": note.is_natural,
                    "origin": note.origin
                }

                results.append(SearchResult(
                    id=note.id,
                    type="note",
                    name=note.name,
                    description=note.description or "",
                    similarity=float(similarity),
                    metadata=metadata
                ))

        except Exception as e:
            print(f"노트 검색 오류: {e}")
            # 폴백: 텍스트 기반 검색
            results = self._fallback_note_search(
                session, filters, top_k
            )

        return results

    def _search_knowledge(
        self,
        session: Session,
        query_embedding: np.ndarray,
        filters: Optional[Dict[str, Any]],
        top_k: int,
        threshold: float
    ) -> List[SearchResult]:
        """지식베이스 검색"""
        results = []

        try:
            query = session.query(
                KnowledgeBase,
                (1 - KnowledgeBase.embedding.cosine_distance(query_embedding)).label("similarity")
            )

            # 필터 적용
            if filters:
                if "category" in filters:
                    query = query.filter(KnowledgeBase.category == filters["category"])
                if "min_confidence" in filters:
                    query = query.filter(KnowledgeBase.confidence >= filters["min_confidence"])

            # 유사도 필터링 및 정렬
            query = query.filter(
                text(f"1 - (embedding <-> :query_embedding) > {threshold}")
            ).params(
                query_embedding=f"[{','.join(map(str, query_embedding))}]"
            ).order_by(
                text("similarity DESC")
            ).limit(top_k)

            for kb, similarity in query:
                metadata = {
                    "category": kb.category,
                    "tags": kb.tags,
                    "confidence": kb.confidence,
                    "source": kb.source
                }

                results.append(SearchResult(
                    id=kb.id,
                    type="knowledge",
                    name=kb.title,
                    description=kb.content,
                    similarity=float(similarity),
                    metadata=metadata
                ))

        except Exception as e:
            print(f"지식베이스 검색 오류: {e}")

        return results

    def search_similar_dna(
        self,
        dna_sequence: List[Tuple[int, float]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """유사한 DNA 시퀀스 검색"""
        # DNA를 텍스트로 변환 후 임베딩
        dna_text = " ".join([f"note_{nid}_{pct}" for nid, pct in dna_sequence])
        dna_embedding = self.encoder.encode(dna_text)

        results = []

        with self.db_manager.get_session() as session:
            try:
                query = session.query(
                    DNASequence,
                    (1 - DNASequence.embedding.cosine_distance(dna_embedding)).label("similarity")
                ).filter(
                    text("1 - (embedding <-> :query_embedding) > 0.5")
                ).params(
                    query_embedding=f"[{','.join(map(str, dna_embedding))}]"
                ).order_by(
                    text("similarity DESC")
                ).limit(top_k)

                for dna, similarity in query:
                    results.append({
                        "id": dna.id,
                        "sequence_id": dna.sequence_id,
                        "genes": dna.genes,
                        "similarity": float(similarity),
                        "scores": {
                            "stability": dna.stability_score,
                            "harmony": dna.harmony_score,
                            "creativity": dna.creativity_score
                        },
                        "generation": dna.generation,
                        "method": dna.generation_method
                    })

            except Exception as e:
                print(f"DNA 검색 오류: {e}")

        return results

    def _fallback_fragrance_search(
        self,
        session: Session,
        query_embedding: np.ndarray,
        filters: Optional[Dict[str, Any]],
        top_k: int
    ) -> List[SearchResult]:
        """향수 폴백 검색 (벡터 검색 실패 시)"""
        results = []

        query = session.query(Fragrance)

        if filters:
            if "family" in filters:
                query = query.filter(Fragrance.family == filters["family"])
            if "gender" in filters:
                query = query.filter(Fragrance.gender == filters["gender"])

        fragrances = query.limit(top_k).all()

        for fragrance in fragrances:
            compositions = session.query(
                Note.name,
                FragranceComposition.percentage
            ).join(
                FragranceComposition
            ).filter(
                FragranceComposition.fragrance_id == fragrance.id
            ).all()

            # 간단한 유사도 계산 (이름 기반)
            similarity = 0.5  # 기본값

            results.append(SearchResult(
                id=fragrance.id,
                type="fragrance",
                name=fragrance.name,
                description=fragrance.description or "",
                similarity=similarity,
                metadata={
                    "family": fragrance.family,
                    "compositions": [
                        {"note": c[0], "percentage": c[1]}
                        for c in compositions
                    ]
                }
            ))

        return results

    def _fallback_note_search(
        self,
        session: Session,
        filters: Optional[Dict[str, Any]],
        top_k: int
    ) -> List[SearchResult]:
        """노트 폴백 검색"""
        results = []

        query = session.query(Note)

        if filters:
            if "type" in filters:
                query = query.filter(Note.type == filters["type"])
            if "pyramid_level" in filters:
                query = query.filter(Note.pyramid_level == filters["pyramid_level"])

        notes = query.limit(top_k).all()

        for note in notes:
            results.append(SearchResult(
                id=note.id,
                type="note",
                name=note.name,
                description=note.description or "",
                similarity=0.5,
                metadata={
                    "type": note.type,
                    "pyramid_level": note.pyramid_level,
                    "volatility": note.volatility
                }
            ))

        return results


# 전역 인스턴스
_search_tool = None


def get_search_tool() -> HybridSearchTool:
    """싱글톤 검색 도구 반환"""
    global _search_tool
    if _search_tool is None:
        _search_tool = HybridSearchTool()
    return _search_tool