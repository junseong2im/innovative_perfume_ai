"""
PostgreSQL 데이터베이스 스키마 정의
pgvector를 사용한 벡터 검색 지원
"""

from sqlalchemy import Column, Integer, String, Float, Text, JSON, DateTime, ForeignKey, Index, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from datetime import datetime

Base = declarative_base()


class Fragrance(Base):
    """향수 마스터 데이터"""
    __tablename__ = 'fragrances'

    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False, unique=True)
    brand = Column(String(100))
    family = Column(String(50))  # floral, woody, citrus, oriental, etc
    gender = Column(String(20))  # masculine, feminine, unisex
    year = Column(Integer)
    description = Column(Text)

    # 벡터 임베딩 (768차원 - sentence-transformers 기준)
    embedding = Column(Vector(768))

    # 메타데이터
    metadata = Column(JSON)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # 관계
    compositions = relationship("FragranceComposition", back_populates="fragrance")
    reviews = relationship("Review", back_populates="fragrance")

    # 인덱스
    __table_args__ = (
        Index('idx_fragrance_embedding', 'embedding', postgresql_using='ivfflat'),
        Index('idx_fragrance_family', 'family'),
        Index('idx_fragrance_brand', 'brand'),
    )


class Note(Base):
    """향료 원재료 데이터"""
    __tablename__ = 'notes'

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, unique=True)
    type = Column(String(50))  # citrus, floral, woody, spicy, etc
    pyramid_level = Column(String(20))  # top, middle, base

    # 화학적 속성
    volatility = Column(Float)  # 0.0 ~ 1.0 (휘발성)
    strength = Column(Float)    # 0.0 ~ 1.0 (강도)
    longevity = Column(Float)   # 0.0 ~ 1.0 (지속성)

    # CAS 번호 (화학물질 식별)
    cas_number = Column(String(20))

    # 자연/합성 여부
    is_natural = Column(Boolean, default=True)

    # 설명
    description = Column(Text)
    origin = Column(String(100))  # 원산지

    # 벡터 임베딩
    embedding = Column(Vector(768))

    # 메타데이터
    metadata = Column(JSON)
    created_at = Column(DateTime, default=func.now())

    # 관계
    compositions = relationship("FragranceComposition", back_populates="note")
    blending_rules = relationship("BlendingRule", foreign_keys="BlendingRule.note1_id")

    __table_args__ = (
        Index('idx_note_embedding', 'embedding', postgresql_using='ivfflat'),
        Index('idx_note_type', 'type'),
        Index('idx_note_pyramid', 'pyramid_level'),
    )


class FragranceComposition(Base):
    """향수 조합 레시피"""
    __tablename__ = 'fragrance_compositions'

    id = Column(Integer, primary_key=True)
    fragrance_id = Column(Integer, ForeignKey('fragrances.id'))
    note_id = Column(Integer, ForeignKey('notes.id'))

    # 비율 (백분율)
    percentage = Column(Float, nullable=False)

    # 피라미드 레벨
    pyramid_level = Column(String(20))  # top, middle, base

    # 관계
    fragrance = relationship("Fragrance", back_populates="compositions")
    note = relationship("Note", back_populates="compositions")

    __table_args__ = (
        Index('idx_composition_fragrance', 'fragrance_id'),
        Index('idx_composition_note', 'note_id'),
    )


class BlendingRule(Base):
    """향료 블렌딩 규칙 (과학적 검증)"""
    __tablename__ = 'blending_rules'

    id = Column(Integer, primary_key=True)
    note1_id = Column(Integer, ForeignKey('notes.id'))
    note2_id = Column(Integer, ForeignKey('notes.id'))

    # 호환성 점수 (-1.0 ~ 1.0)
    compatibility = Column(Float, nullable=False)

    # 규칙 유형
    rule_type = Column(String(50))  # harmony, conflict, enhancement, etc

    # 설명
    description = Column(Text)

    # 과학적 근거
    scientific_basis = Column(Text)

    # 검증 여부
    is_verified = Column(Boolean, default=False)

    # 관계
    note1 = relationship("Note", foreign_keys=[note1_id])
    note2 = relationship("Note", foreign_keys=[note2_id])

    __table_args__ = (
        Index('idx_blending_notes', 'note1_id', 'note2_id'),
    )


class AccordTemplate(Base):
    """어코드 템플릿 (향수 베이스 구조)"""
    __tablename__ = 'accord_templates'

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    type = Column(String(50))  # fougere, chypre, oriental, etc

    # 구조 정의 (JSON)
    structure = Column(JSON, nullable=False)
    # 예: {"top": [{"note": "bergamot", "min": 5, "max": 15}], ...}

    # 설명
    description = Column(Text)
    historical_context = Column(Text)

    # 사용 빈도
    usage_count = Column(Integer, default=0)

    # 벡터 임베딩
    embedding = Column(Vector(768))

    created_at = Column(DateTime, default=func.now())

    __table_args__ = (
        Index('idx_accord_embedding', 'embedding', postgresql_using='ivfflat'),
    )


class Review(Base):
    """향수 리뷰 데이터"""
    __tablename__ = 'reviews'

    id = Column(Integer, primary_key=True)
    fragrance_id = Column(Integer, ForeignKey('fragrances.id'))
    user_id = Column(String(100))

    # 평점
    rating = Column(Float)

    # 리뷰 텍스트
    review_text = Column(Text)

    # 감정 분석 결과
    sentiment_score = Column(Float)  # -1.0 ~ 1.0

    # 특징 추출
    detected_notes = Column(JSON)  # 리뷰에서 언급된 노트들
    mood_keywords = Column(JSON)   # 추출된 무드 키워드

    # 벡터 임베딩
    embedding = Column(Vector(768))

    created_at = Column(DateTime, default=func.now())

    # 관계
    fragrance = relationship("Fragrance", back_populates="reviews")

    __table_args__ = (
        Index('idx_review_embedding', 'embedding', postgresql_using='ivfflat'),
        Index('idx_review_fragrance', 'fragrance_id'),
    )


class KnowledgeBase(Base):
    """향수 지식베이스"""
    __tablename__ = 'knowledge_base'

    id = Column(Integer, primary_key=True)
    category = Column(String(50), nullable=False)  # history, technique, ingredient, etc
    title = Column(String(200), nullable=False)
    content = Column(Text, nullable=False)

    # 관련 태그
    tags = Column(JSON)

    # 신뢰도 점수
    confidence = Column(Float, default=1.0)

    # 출처
    source = Column(String(200))

    # 벡터 임베딩
    embedding = Column(Vector(768))

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index('idx_knowledge_embedding', 'embedding', postgresql_using='ivfflat'),
        Index('idx_knowledge_category', 'category'),
    )


class DNASequence(Base):
    """AI 생성 향수 DNA"""
    __tablename__ = 'dna_sequences'

    id = Column(Integer, primary_key=True)
    sequence_id = Column(String(100), unique=True, nullable=False)

    # DNA 유전자 (향료 ID와 비율)
    genes = Column(JSON, nullable=False)
    # 예: [{"note_id": 1, "percentage": 10.5}, ...]

    # 피트니스 점수들
    stability_score = Column(Float)
    harmony_score = Column(Float)
    creativity_score = Column(Float)

    # 생성 방법
    generation_method = Column(String(50))  # moga, rlhf, manual

    # 부모 DNA (진화된 경우)
    parent_id = Column(Integer, ForeignKey('dna_sequences.id'))
    generation = Column(Integer, default=1)

    # 사용자 피드백
    user_feedback = Column(JSON)

    # 벡터 임베딩
    embedding = Column(Vector(768))

    created_at = Column(DateTime, default=func.now())

    # 관계
    parent = relationship("DNASequence", remote_side=[id])

    __table_args__ = (
        Index('idx_dna_embedding', 'embedding', postgresql_using='ivfflat'),
        Index('idx_dna_sequence', 'sequence_id'),
    )


class ModelWeights(Base):
    """학습된 모델 가중치 저장"""
    __tablename__ = 'model_weights'

    id = Column(Integer, primary_key=True)
    model_name = Column(String(100), nullable=False)
    model_type = Column(String(50))  # validator, generator, policy_network

    # 가중치 (바이너리 또는 경로)
    weights_path = Column(String(500))

    # 메타데이터
    architecture = Column(JSON)
    hyperparameters = Column(JSON)
    training_metrics = Column(JSON)

    # 버전 관리
    version = Column(String(20))
    is_active = Column(Boolean, default=True)

    created_at = Column(DateTime, default=func.now())

    __table_args__ = (
        Index('idx_model_name', 'model_name'),
        Index('idx_model_active', 'is_active'),
    )