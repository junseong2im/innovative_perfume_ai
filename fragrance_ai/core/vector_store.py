from typing import List, Dict, Any, Optional, Tuple, Union
import chromadb
from chromadb.config import Settings as ChromaSettings
import numpy as np
import logging
import asyncio
from datetime import datetime
import uuid

from .config import settings

logger = logging.getLogger(__name__)

class VectorStore:
    """ChromaDB 기반 벡터 스토어 클래스"""
    
    def __init__(self):
        """Enhanced 벡터 스토어 초기화"""
        self.client = None
        self.collections = {}
        self.embedding_function = None

        # Performance optimization settings
        self.batch_size = 1000  # 기본 배치 크기
        self.max_batch_size = 5000  # 최대 배치 크기
        self.index_refresh_interval = 100  # 인덱스 갱신 간격
        
    async def initialize(self):
        """비동기 초기화"""
        try:
            # ChromaDB 클라이언트 초기화
            self.client = chromadb.PersistentClient(
                path=settings.chroma_persist_directory,
                settings=ChromaSettings(
                    allow_reset=True,
                    anonymized_telemetry=False,
                    is_persistent=True
                )
            )
            
            # 기본 컬렉션들 초기화
            await self._initialize_collections()
            logger.info("VectorStore initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize VectorStore: {e}")
            raise

    async def add_documents_batch(
        self,
        collection_name: str,
        documents: List[Dict[str, Any]],
        batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        대용량 문서를 효율적으로 배치 삽입

        Args:
            collection_name: 컬렉션 이름
            documents: 문서 리스트
            batch_size: 배치 크기 (None이면 자동 계산)
        """
        try:
            # 자동 배치 크기 계산
            if batch_size is None:
                batch_size = min(self.batch_size, len(documents))

            total_documents = len(documents)
            processed = 0
            results = []

            logger.info(f"Starting batch insertion of {total_documents} documents with batch size {batch_size}")

            # 배치별로 처리
            for i in range(0, total_documents, batch_size):
                batch_docs = documents[i:i + batch_size]

                try:
                    result = await self.add_documents(collection_name, batch_docs)
                    results.append(result)
                    processed += len(batch_docs)

                    if processed % (batch_size * 5) == 0:  # 5배치마다 로그
                        logger.info(f"Processed {processed}/{total_documents} documents ({processed/total_documents*100:.1f}%)")

                except Exception as e:
                    logger.error(f"Failed to process batch {i//batch_size + 1}: {e}")
                    # 배치 크기를 줄여서 재시도
                    if batch_size > 10:
                        smaller_batch_size = batch_size // 2
                        logger.info(f"Retrying with smaller batch size: {smaller_batch_size}")
                        await self.add_documents_batch(collection_name, batch_docs, smaller_batch_size)

            logger.info(f"Batch insertion completed: {processed} documents processed")
            return {
                "status": "success",
                "total_processed": processed,
                "batches_processed": len(results),
                "batch_size": batch_size
            }

        except Exception as e:
            logger.error(f"Failed to batch insert documents: {e}")
            raise

    async def _initialize_collections(self):
        """기본 컬렉션들 초기화"""
        collection_configs = {
            "fragrance_notes": {
                "description": "향료 노트 정보",
                "metadata": {"type": "fragrance_note", "language": "ko"}
            },
            "recipes": {
                "description": "향수 레시피",
                "metadata": {"type": "recipe", "language": "ko"}
            },
            "brands": {
                "description": "향수 브랜드 정보",
                "metadata": {"type": "brand", "language": "ko"}
            },
            "mood_descriptions": {
                "description": "감성 및 무드 설명",
                "metadata": {"type": "mood", "language": "ko"}
            }
        }
        
        for collection_name, config in collection_configs.items():
            try:
                collection = self.client.get_or_create_collection(
                    name=collection_name,
                    metadata=config["metadata"]
                )
                self.collections[collection_name] = collection
                logger.info(f"Initialized collection: {collection_name}")
            except Exception as e:
                logger.error(f"Failed to initialize collection {collection_name}: {e}")
    
    async def add_documents(
        self, 
        collection_name: str, 
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        문서들을 컬렉션에 추가
        
        Args:
            collection_name: 컬렉션 이름
            documents: 문서 리스트 [{"id": str, "document": str, "embedding": List[float], "metadata": dict}]
        """
        try:
            collection = await self._get_collection(collection_name)
            
            ids = []
            embeddings = []
            docs = []
            metadatas = []
            
            for doc in documents:
                doc_id = doc.get("id", str(uuid.uuid4()))
                document_text = doc.get("document", "")
                embedding = doc.get("embedding")
                metadata = doc.get("metadata", {})
                
                # 메타데이터에 타임스탬프 추가
                metadata.update({
                    "created_at": datetime.utcnow().isoformat(),
                    "document_length": len(document_text)
                })
                
                ids.append(doc_id)
                docs.append(document_text)
                embeddings.append(embedding)
                metadatas.append(metadata)
            
            # ChromaDB에 추가
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=docs,
                metadatas=metadatas
            )
            
            logger.info(f"Added {len(documents)} documents to collection '{collection_name}'")
            
            return {
                "success": True,
                "collection": collection_name,
                "documents_added": len(documents),
                "document_ids": ids
            }
            
        except Exception as e:
            logger.error(f"Failed to add documents to collection '{collection_name}': {e}")
            raise
    
    async def search(
        self,
        collection_name: str,
        query_embedding: List[float],
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        similarity_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        벡터 검색 수행
        
        Args:
            collection_name: 검색할 컬렉션 이름
            query_embedding: 쿼리 임베딩 벡터
            n_results: 반환할 결과 수
            where: 필터 조건
            similarity_threshold: 유사도 임계값
        """
        try:
            collection = await self._get_collection(collection_name)
            
            # ChromaDB 검색 수행
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"]
            )
            
            # 결과 포맷팅
            formatted_results = []
            
            if results["ids"] and len(results["ids"]) > 0:
                ids = results["ids"][0]
                documents = results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results["distances"][0]
                
                for i in range(len(ids)):
                    # 유사도 계산 (거리를 유사도로 변환)
                    similarity = 1 - distances[i]
                    
                    # 임계값 필터링
                    if similarity >= similarity_threshold:
                        formatted_results.append({
                            "id": ids[i],
                            "document": documents[i],
                            "metadata": metadatas[i],
                            "distance": distances[i],
                            "similarity": similarity
                        })
            
            logger.info(f"Search completed: {len(formatted_results)} results from '{collection_name}'")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed in collection '{collection_name}': {e}")
            raise
    
    async def update_documents(
        self,
        collection_name: str,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """문서들 업데이트"""
        try:
            collection = await self._get_collection(collection_name)
            
            ids = []
            embeddings = []
            docs = []
            metadatas = []
            
            for doc in documents:
                doc_id = doc.get("id")
                if not doc_id:
                    raise ValueError("Document ID is required for update")
                
                document_text = doc.get("document", "")
                embedding = doc.get("embedding")
                metadata = doc.get("metadata", {})
                
                # 메타데이터에 업데이트 타임스탬프 추가
                metadata.update({
                    "updated_at": datetime.utcnow().isoformat(),
                    "document_length": len(document_text)
                })
                
                ids.append(doc_id)
                docs.append(document_text)
                embeddings.append(embedding)
                metadatas.append(metadata)
            
            # ChromaDB 업데이트
            collection.update(
                ids=ids,
                embeddings=embeddings,
                documents=docs,
                metadatas=metadatas
            )
            
            logger.info(f"Updated {len(documents)} documents in collection '{collection_name}'")
            
            return {
                "success": True,
                "collection": collection_name,
                "documents_updated": len(documents),
                "document_ids": ids
            }
            
        except Exception as e:
            logger.error(f"Failed to update documents in collection '{collection_name}': {e}")
            raise
    
    async def delete_documents(
        self,
        collection_name: str,
        document_ids: List[str]
    ) -> Dict[str, Any]:
        """문서들 삭제"""
        try:
            collection = await self._get_collection(collection_name)
            
            collection.delete(ids=document_ids)
            
            logger.info(f"Deleted {len(document_ids)} documents from collection '{collection_name}'")
            
            return {
                "success": True,
                "collection": collection_name,
                "documents_deleted": len(document_ids),
                "document_ids": document_ids
            }
            
        except Exception as e:
            logger.error(f"Failed to delete documents from collection '{collection_name}': {e}")
            raise
    
    async def get_item(
        self,
        collection_name: str,
        item_id: str
    ) -> Optional[Dict[str, Any]]:
        """특정 아이템 조회"""
        try:
            collection = await self._get_collection(collection_name)
            
            results = collection.get(
                ids=[item_id],
                include=["documents", "metadatas", "embeddings"]
            )
            
            if results["ids"] and len(results["ids"]) > 0:
                return {
                    "id": results["ids"][0],
                    "document": results["documents"][0],
                    "metadata": results["metadatas"][0],
                    "embedding": results["embeddings"][0] if results["embeddings"] else None
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get item '{item_id}' from collection '{collection_name}': {e}")
            raise
    
    async def get_all_documents(
        self,
        collection_name: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """컬렉션의 모든 문서 조회"""
        try:
            collection = await self._get_collection(collection_name)
            
            # ChromaDB get 메서드로 모든 문서 조회
            results = collection.get(
                limit=limit,
                include=["documents", "metadatas", "embeddings"]
            )
            
            documents = []
            if results["ids"]:
                for i in range(len(results["ids"])):
                    documents.append({
                        "id": results["ids"][i],
                        "document": results["documents"][i],
                        "metadata": results["metadatas"][i],
                        "embedding": results["embeddings"][i] if results["embeddings"] else None
                    })
            
            logger.info(f"Retrieved {len(documents)} documents from collection '{collection_name}'")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to get all documents from collection '{collection_name}': {e}")
            raise
    
    async def get_collection_info(
        self,
        collection_name: str
    ) -> Dict[str, Any]:
        """컬렉션 정보 조회"""
        try:
            collection = await self._get_collection(collection_name)
            
            # 컬렉션 통계 수집
            count = collection.count()
            
            # 샘플 문서로부터 메타데이터 필드 추출
            sample_docs = collection.get(limit=5, include=["metadatas"])
            metadata_fields = set()
            
            if sample_docs["metadatas"]:
                for metadata in sample_docs["metadatas"]:
                    if metadata:
                        metadata_fields.update(metadata.keys())
            
            return {
                "name": collection_name,
                "count": count,
                "metadata_fields": list(metadata_fields),
                "dimension": settings.vector_dimension,  # 설정에서 가져옴
                "last_updated": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection info for '{collection_name}': {e}")
            raise
    
    async def list_collections(self) -> List[Dict[str, Any]]:
        """모든 컬렉션 목록 조회"""
        try:
            collections = self.client.list_collections()
            
            collection_infos = []
            for collection in collections:
                info = await self.get_collection_info(collection.name)
                collection_infos.append(info)
            
            return collection_infos
            
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            raise
    
    async def create_collection(
        self,
        collection_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """새 컬렉션 생성"""
        try:
            if metadata is None:
                metadata = {}
            
            metadata.update({
                "created_at": datetime.utcnow().isoformat(),
                "type": "user_created"
            })
            
            collection = self.client.create_collection(
                name=collection_name,
                metadata=metadata
            )
            
            self.collections[collection_name] = collection
            
            logger.info(f"Created collection: {collection_name}")
            
            return {
                "success": True,
                "collection_name": collection_name,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to create collection '{collection_name}': {e}")
            raise
    
    async def delete_collection(self, collection_name: str) -> Dict[str, Any]:
        """컬렉션 삭제"""
        try:
            self.client.delete_collection(name=collection_name)
            
            if collection_name in self.collections:
                del self.collections[collection_name]
            
            logger.info(f"Deleted collection: {collection_name}")
            
            return {
                "success": True,
                "collection_name": collection_name
            }
            
        except Exception as e:
            logger.error(f"Failed to delete collection '{collection_name}': {e}")
            raise
    
    async def batch_search(
        self,
        collection_names: List[str],
        query_embedding: List[float],
        n_results: int = 10,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """여러 컬렉션에서 동시 검색"""
        search_tasks = []
        
        for collection_name in collection_names:
            task = self.search(
                collection_name=collection_name,
                query_embedding=query_embedding,
                n_results=n_results
            )
            search_tasks.append((collection_name, task))
        
        # 모든 검색 작업을 병렬로 실행
        results = {}
        for collection_name, task in search_tasks:
            try:
                collection_results = await task
                
                # 가중치 적용 (제공된 경우)
                if weights and collection_name in weights:
                    weight = weights[collection_name]
                    for result in collection_results:
                        result["weighted_similarity"] = result["similarity"] * weight
                        result["original_similarity"] = result["similarity"]
                        result["similarity"] = result["weighted_similarity"]
                
                results[collection_name] = collection_results
            except Exception as e:
                logger.warning(f"Search failed for collection '{collection_name}': {e}")
                results[collection_name] = []
        
        return results
    
    async def _get_collection(self, collection_name: str):
        """컬렉션 객체 조회"""
        if collection_name not in self.collections:
            try:
                collection = self.client.get_collection(name=collection_name)
                self.collections[collection_name] = collection
            except Exception as e:
                # 컬렉션이 존재하지 않으면 생성
                logger.warning(f"Collection '{collection_name}' not found, creating new one")
                await self.create_collection(collection_name)
                collection = self.client.get_collection(name=collection_name)
                self.collections[collection_name] = collection
        
        return self.collections[collection_name]
    
    async def health_check(self) -> Dict[str, Any]:
        """벡터 스토어 상태 확인"""
        try:
            # 기본 연결 테스트
            collections = self.client.list_collections()
            
            # 각 컬렉션 상태 확인
            collection_status = {}
            for collection in collections:
                try:
                    count = collection.count()
                    collection_status[collection.name] = {
                        "status": "healthy",
                        "document_count": count
                    }
                except Exception as e:
                    collection_status[collection.name] = {
                        "status": "error",
                        "error": str(e)
                    }
            
            return {
                "status": "healthy",
                "total_collections": len(collections),
                "collections": collection_status,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }