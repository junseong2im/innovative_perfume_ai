from typing import List, Dict, Any, Optional, Tuple
import logging
import time
from datetime import datetime
import json

from ..core.vector_store import VectorStore
from ..models.embedding import AdvancedKoreanFragranceEmbedding
from ..core.config import settings

logger = logging.getLogger(__name__)

class SearchService:
    """향수 검색을 위한 통합 서비스"""
    
    def __init__(self):
        self.vector_store = VectorStore()
        self.embedding_model = AdvancedKoreanFragranceEmbedding()
        self.search_cache = {}
        self.cache_ttl = 3600  # 1시간
        
    async def initialize(self):
        """서비스 초기화"""
        try:
            await self.vector_store.initialize()
            await self.embedding_model.initialize()
            logger.info("SearchService initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize SearchService: {e}")
            raise
    
    async def semantic_search(
        self,
        query: str,
        collection_names: Optional[List[str]] = None,
        search_type: str = "similarity",
        top_k: int = 10,
        similarity_threshold: float = 0.7,
        collection_weights: Optional[Dict[str, float]] = None,
        filters: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        의미 기반 검색 수행
        
        Args:
            query: 검색 쿼리
            collection_names: 검색할 컬렉션 목록
            search_type: 검색 타입 (similarity, hybrid, single_collection)
            top_k: 반환할 결과 수
            similarity_threshold: 유사도 임계값
            collection_weights: 컬렉션별 가중치
            filters: 추가 필터
            include_metadata: 메타데이터 포함 여부
            use_cache: 캐시 사용 여부
        """
        start_time = time.time()
        
        # 캐시 확인
        if use_cache:
            cache_key = self._generate_cache_key(
                query, collection_names, search_type, top_k, similarity_threshold
            )
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                logger.info(f"Cache hit for query: {query[:50]}...")
                return cached_result
        
        try:
            # 쿼리 임베딩 생성
            query_embedding = await self.embedding_model.encode_query(query)
            
            # 기본 컬렉션 설정
            if not collection_names:
                collection_names = ["fragrance_notes", "recipes", "brands"]
            
            # 컬렉션별 가중치 기본값
            if not collection_weights:
                collection_weights = {
                    "fragrance_notes": 1.0,
                    "recipes": 0.8,
                    "brands": 0.6
                }
            
            # 검색 수행
            if search_type == "single_collection":
                results = await self._single_collection_search(
                    query_embedding, collection_names[0], top_k, similarity_threshold, filters
                )
            elif search_type == "hybrid":
                results = await self._hybrid_search(
                    query_embedding, collection_names, top_k, similarity_threshold,
                    collection_weights, filters
                )
            else:  # similarity
                results = await self._similarity_search(
                    query_embedding, collection_names, top_k, similarity_threshold, filters
                )
            
            # 결과 후처리
            processed_results = await self._process_search_results(
                results, query, include_metadata
            )
            
            # 검색 시간 계산
            search_time = time.time() - start_time
            
            # 최종 응답 구성
            response = {
                "query": query,
                "results": processed_results,
                "metadata": {
                    "total_results": len(processed_results),
                    "search_time": round(search_time, 3),
                    "search_type": search_type,
                    "collections_searched": collection_names,
                    "similarity_threshold": similarity_threshold,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
            # 결과 캐싱
            if use_cache and cache_key:
                self._cache_result(cache_key, response)
            
            logger.info(f"Search completed: {len(processed_results)} results in {search_time:.3f}s")
            return response
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    async def _single_collection_search(
        self, 
        query_embedding: List[float], 
        collection_name: str, 
        top_k: int, 
        similarity_threshold: float,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """단일 컬렉션 검색"""
        return await self.vector_store.search(
            collection_name=collection_name,
            query_embedding=query_embedding,
            n_results=top_k,
            where=filters,
            similarity_threshold=similarity_threshold
        )
    
    async def _similarity_search(
        self,
        query_embedding: List[float],
        collection_names: List[str],
        top_k: int,
        similarity_threshold: float,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """유사도 기반 검색"""
        all_results = []
        
        for collection_name in collection_names:
            try:
                results = await self.vector_store.search(
                    collection_name=collection_name,
                    query_embedding=query_embedding,
                    n_results=top_k,
                    where=filters,
                    similarity_threshold=similarity_threshold
                )
                
                # 컬렉션 정보 추가
                for result in results:
                    result["collection"] = collection_name
                
                all_results.extend(results)
            except Exception as e:
                logger.warning(f"Search failed for collection {collection_name}: {e}")
                continue
        
        # 유사도 기준으로 정렬 및 상위 결과 반환
        all_results.sort(key=lambda x: x.get("distance", 1.0))
        return all_results[:top_k]
    
    async def _hybrid_search(
        self,
        query_embedding: List[float],
        collection_names: List[str],
        top_k: int,
        similarity_threshold: float,
        collection_weights: Dict[str, float],
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """하이브리드 검색 (가중치 적용)"""
        weighted_results = []
        
        for collection_name in collection_names:
            try:
                results = await self.vector_store.search(
                    collection_name=collection_name,
                    query_embedding=query_embedding,
                    n_results=top_k,
                    where=filters,
                    similarity_threshold=similarity_threshold
                )
                
                # 가중치 적용
                weight = collection_weights.get(collection_name, 1.0)
                for result in results:
                    result["collection"] = collection_name
                    result["original_distance"] = result.get("distance", 1.0)
                    result["weighted_score"] = (1 - result.get("distance", 1.0)) * weight
                    result["distance"] = 1 - result["weighted_score"]
                
                weighted_results.extend(results)
            except Exception as e:
                logger.warning(f"Hybrid search failed for collection {collection_name}: {e}")
                continue
        
        # 가중 점수 기준으로 정렬
        weighted_results.sort(key=lambda x: x.get("weighted_score", 0), reverse=True)
        return weighted_results[:top_k]
    
    async def _process_search_results(
        self, 
        results: List[Dict[str, Any]], 
        query: str,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """검색 결과 후처리"""
        processed_results = []
        
        for result in results:
            processed_result = {
                "id": result.get("id"),
                "document": result.get("document"),
                "similarity_score": 1 - result.get("distance", 1.0),
                "collection": result.get("collection")
            }
            
            if include_metadata:
                metadata = result.get("metadata", {})
                processed_result["metadata"] = metadata
                
                # 향수 노트 특화 메타데이터 처리
                if result.get("collection") == "fragrance_notes":
                    processed_result["note_type"] = metadata.get("note_type")
                    processed_result["intensity"] = metadata.get("intensity")
                    processed_result["mood_tags"] = metadata.get("mood_tags", [])
                
                # 레시피 특화 메타데이터 처리
                elif result.get("collection") == "recipes":
                    processed_result["recipe_type"] = metadata.get("recipe_type")
                    processed_result["complexity"] = metadata.get("complexity")
                    processed_result["ingredients_count"] = metadata.get("ingredients_count")
            
            processed_results.append(processed_result)
        
        return processed_results
    
    def _generate_cache_key(
        self, 
        query: str, 
        collection_names: Optional[List[str]], 
        search_type: str, 
        top_k: int, 
        similarity_threshold: float
    ) -> str:
        """캐시 키 생성"""
        import hashlib
        
        cache_data = {
            "query": query.lower().strip(),
            "collections": sorted(collection_names or []),
            "search_type": search_type,
            "top_k": top_k,
            "similarity_threshold": similarity_threshold
        }
        
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """캐시된 결과 조회"""
        if cache_key in self.search_cache:
            cached_item = self.search_cache[cache_key]
            if time.time() - cached_item["timestamp"] < self.cache_ttl:
                return cached_item["result"]
            else:
                # 만료된 캐시 삭제
                del self.search_cache[cache_key]
        return None
    
    def _cache_result(self, cache_key: str, result: Dict[str, Any]):
        """결과 캐싱"""
        self.search_cache[cache_key] = {
            "result": result,
            "timestamp": time.time()
        }
        
        # 캐시 크기 제한 (최대 1000개 항목)
        if len(self.search_cache) > 1000:
            # 가장 오래된 항목 삭제
            oldest_key = min(self.search_cache.keys(), 
                           key=lambda k: self.search_cache[k]["timestamp"])
            del self.search_cache[oldest_key]
    
    async def get_similar_items(
        self, 
        item_id: str, 
        collection_name: str, 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """특정 아이템과 유사한 항목 찾기"""
        try:
            # 아이템 정보 조회
            item_data = await self.vector_store.get_item(collection_name, item_id)
            if not item_data:
                raise ValueError(f"Item {item_id} not found in collection {collection_name}")
            
            # 해당 아이템의 임베딩으로 유사 검색
            item_embedding = item_data.get("embedding")
            if not item_embedding:
                raise ValueError(f"No embedding found for item {item_id}")
            
            # 자기 자신 제외하고 검색
            results = await self.vector_store.search(
                collection_name=collection_name,
                query_embedding=item_embedding,
                n_results=top_k + 1,  # 자기 자신 포함되므로 +1
                where={"id": {"$ne": item_id}}  # 자기 자신 제외
            )
            
            return await self._process_search_results(results[:top_k], "", True)
            
        except Exception as e:
            logger.error(f"Failed to find similar items: {e}")
            raise
    
    async def add_fragrance_data(
        self, 
        collection_name: str, 
        data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """향수 데이터 배치 추가"""
        try:
            start_time = time.time()
            
            # 문서들의 임베딩 생성
            documents = [item["document"] for item in data]
            embeddings = await self.embedding_model.encode_batch(documents)
            
            # 임베딩과 메타데이터 결합
            processed_data = []
            for i, item in enumerate(data):
                processed_item = {
                    "id": item["id"],
                    "document": item["document"],
                    "embedding": embeddings[i],
                    "metadata": item.get("metadata", {})
                }
                processed_data.append(processed_item)
            
            # 벡터 스토어에 추가
            result = await self.vector_store.add_documents(collection_name, processed_data)
            
            processing_time = time.time() - start_time
            logger.info(f"Added {len(data)} items to {collection_name} in {processing_time:.3f}s")
            
            return {
                "success": True,
                "items_added": len(data),
                "processing_time": round(processing_time, 3),
                "collection": collection_name
            }
            
        except Exception as e:
            logger.error(f"Failed to add fragrance data: {e}")
            raise
    
    async def update_embeddings(self, collection_name: str) -> Dict[str, Any]:
        """컬렉션의 모든 임베딩 재생성"""
        try:
            start_time = time.time()
            
            # 모든 문서 조회
            all_documents = await self.vector_store.get_all_documents(collection_name)
            
            if not all_documents:
                return {"success": True, "message": "No documents to update"}
            
            # 임베딩 재생성
            documents = [doc["document"] for doc in all_documents]
            new_embeddings = await self.embedding_model.encode_batch(documents)
            
            # 업데이트된 데이터 준비
            updated_data = []
            for i, doc in enumerate(all_documents):
                doc["embedding"] = new_embeddings[i]
                updated_data.append(doc)
            
            # 벡터 스토어 업데이트
            await self.vector_store.update_documents(collection_name, updated_data)
            
            processing_time = time.time() - start_time
            logger.info(f"Updated embeddings for {len(updated_data)} items in {processing_time:.3f}s")
            
            return {
                "success": True,
                "items_updated": len(updated_data),
                "processing_time": round(processing_time, 3),
                "collection": collection_name
            }
            
        except Exception as e:
            logger.error(f"Failed to update embeddings: {e}")
            raise
    
    async def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """컬렉션 통계 정보"""
        try:
            stats = await self.vector_store.get_collection_info(collection_name)
            return {
                "collection_name": collection_name,
                "document_count": stats.get("count", 0),
                "last_updated": stats.get("last_updated"),
                "embedding_dimension": stats.get("dimension"),
                "index_size": stats.get("index_size"),
                "metadata_fields": stats.get("metadata_fields", [])
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            raise
    
    async def find_similar_items(self, item_id: str, collection_name: str, top_k: int = 5) -> Dict[str, Any]:
        """특정 아이템과 유사한 아이템 찾기"""
        try:
            # 먼저 해당 아이템을 찾아 임베딩을 가져옴
            item_data = await self.vector_store.get_by_id(collection_name, item_id)
            if not item_data:
                raise ValueError(f"Item {item_id} not found in collection {collection_name}")
            
            # 해당 아이템의 임베딩을 사용해 유사한 아이템 검색
            if "embedding" in item_data:
                similar_results = await self.vector_store.similarity_search_by_vector(
                    collection_name=collection_name,
                    query_vector=item_data["embedding"],
                    top_k=top_k + 1,  # 자기 자신 제외를 위해 +1
                    filters={"id": {"$ne": item_id}}  # 자기 자신 제외
                )
            else:
                # 임베딩이 없으면 텍스트 기반 검색
                similar_results = await self.semantic_search(
                    query=item_data.get("content", ""),
                    collection_name=collection_name,
                    top_k=top_k + 1,
                    filters={"id": {"$ne": item_id}}
                )
            
            similar_items = []
            for result in similar_results.get("results", [])[:top_k]:
                if result.get("id") != item_id:  # 혹시 모를 중복 제거
                    similar_items.append({
                        "id": result.get("id"),
                        "content": result.get("document", result.get("content")),
                        "metadata": result.get("metadata", {}),
                        "similarity": 1.0 - result.get("distance", 1.0)
                    })
            
            return {
                "similar_items": similar_items,
                "total_found": len(similar_items),
                "reference_item": {
                    "id": item_id,
                    "content": item_data.get("content", ""),
                    "metadata": item_data.get("metadata", {})
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to find similar items for {item_id}: {e}")
            raise
    
    async def clear_collection(self, collection_name: str) -> int:
        """컬렉션의 모든 문서 삭제"""
        try:
            # 삭제하기 전 문서 수 확인
            stats = await self.get_collection_stats(collection_name)
            document_count = stats.get("document_count", 0)
            
            # 컬렉션 삭제
            await self.vector_store.delete_collection(collection_name)
            
            logger.info(f"Cleared collection {collection_name} with {document_count} documents")
            return document_count
            
        except Exception as e:
            logger.error(f"Failed to clear collection {collection_name}: {e}")
            raise
    
    async def add_document(self, collection_name: str, document: Dict[str, Any]):
        """단일 문서 추가"""
        try:
            # 문서 임베딩 생성
            content = document.get("content", "")
            if content:
                embedding = await self.embedding_model.encode_async(content)
                document["embedding"] = embedding
            
            # 벡터 스토어에 추가
            await self.vector_store.add_documents(
                collection_name=collection_name,
                documents=[document]
            )
            
            logger.info(f"Added document {document.get('id', 'unknown')} to {collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to add document to {collection_name}: {e}")
            raise
    
    async def search_collection(self, collection_name: str, query: str, top_k: int = 10, min_similarity: float = 0.5) -> Dict[str, Any]:
        """특정 컬렉션에서 검색"""
        try:
            start_time = time.time()
            
            # 의미 기반 검색 실행
            search_results = await self.semantic_search(
                query=query,
                collection_name=collection_name,
                top_k=top_k,
                min_similarity=min_similarity
            )
            
            # 검색 시간 추가
            search_results["search_time"] = time.time() - start_time
            
            return search_results
            
        except Exception as e:
            logger.error(f"Failed to search collection {collection_name}: {e}")
            raise
    
    async def reload_models(self):
        """모델 재로드"""
        try:
            # 임베딩 모델 재로드
            await self.embedding_model.initialize()
            
            # 벡터 스토어 재연결
            await self.vector_store.initialize()
            
            logger.info("Successfully reloaded search service models")
            
        except Exception as e:
            logger.error(f"Failed to reload search service models: {e}")
            raise