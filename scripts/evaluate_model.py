#!/usr/bin/env python3
"""
향수 AI 모델 평가 스크립트
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fragrance_ai.models.generator import FragranceGenerator
from fragrance_ai.models.embedding import FragranceEmbedding
from fragrance_ai.services.search_service import SearchService
from fragrance_ai.services.generation_service import GenerationService
from fragrance_ai.evaluation.metrics import EvaluationMetrics, QualityAssessment
from fragrance_ai.utils.data_loader import DatasetLoader

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(description="향수 AI 모델 평가")
    
    # 기본 설정
    parser.add_argument("--model-type", type=str, 
                       choices=["embedding", "generation", "search", "end-to-end"],
                       required=True, help="평가할 모델 타입")
    parser.add_argument("--model-path", type=str, required=True, help="모델 경로")
    parser.add_argument("--eval-data", type=str, required=True, help="평가 데이터 경로")
    parser.add_argument("--output-dir", type=str, default="./evaluation_results", 
                       help="결과 저장 디렉토리")
    
    # 평가 설정
    parser.add_argument("--batch-size", type=int, default=16, help="배치 크기")
    parser.add_argument("--max-samples", type=int, help="최대 평가 샘플 수")
    parser.add_argument("--k-values", nargs='+', type=int, default=[1, 5, 10, 20], 
                       help="검색 평가를 위한 K 값들")
    
    # 생성 평가 설정
    parser.add_argument("--generation-config", type=str, help="생성 설정 파일")
    parser.add_argument("--temperature", type=float, default=0.7, help="생성 온도")
    parser.add_argument("--max-tokens", type=int, default=800, help="최대 토큰 수")
    
    # 기타 설정
    parser.add_argument("--save-results", action="store_true", help="개별 결과 저장")
    parser.add_argument("--compare-with", type=str, help="비교할 기준 모델 경로")
    parser.add_argument("--verbose", action="store_true", help="상세 로그 출력")
    
    return parser.parse_args()

class ModelEvaluator:
    """모델 평가 클래스"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.quality_assessor = QualityAssessment()
        os.makedirs(output_dir, exist_ok=True)
        
    def evaluate_embedding_model(
        self, 
        model_path: str, 
        eval_data_path: str,
        batch_size: int = 16,
        max_samples: Optional[int] = None,
        k_values: List[int] = [1, 5, 10, 20]
    ) -> Dict[str, Any]:
        """임베딩 모델 평가"""
        logger.info("임베딩 모델 평가 시작")
        
        # 모델 로드
        embedding_model = FragranceEmbedding()
        embedding_model.load_model(model_path)
        
        # 평가 데이터 로드
        data_loader = DatasetLoader()
        eval_data = data_loader.load_embedding_eval_dataset(eval_data_path)
        
        if max_samples:
            eval_data = eval_data[:max_samples]
        
        logger.info(f"평가 데이터 크기: {len(eval_data)}")
        
        # 쿼리와 문서 분리
        queries = [item["query"] for item in eval_data]
        documents = [item["document"] for item in eval_data]
        relevance_labels = np.array([item["relevance"] for item in eval_data])
        
        # 임베딩 생성
        logger.info("쿼리 임베딩 생성 중...")
        query_embeddings = []
        for i in tqdm(range(0, len(queries), batch_size)):
            batch_queries = queries[i:i+batch_size]
            batch_embeddings = embedding_model.encode_batch(batch_queries)
            query_embeddings.extend(batch_embeddings)
        
        logger.info("문서 임베딩 생성 중...")
        document_embeddings = []
        for i in tqdm(range(0, len(documents), batch_size)):
            batch_documents = documents[i:i+batch_size]
            batch_embeddings = embedding_model.encode_batch(batch_documents)
            document_embeddings.extend(batch_embeddings)
        
        query_embeddings = np.array(query_embeddings)
        document_embeddings = np.array(document_embeddings)
        
        # 메트릭 계산
        logger.info("메트릭 계산 중...")
        metrics = EvaluationMetrics.calculate_embedding_metrics(
            query_embeddings=query_embeddings,
            document_embeddings=document_embeddings,
            relevance_labels=relevance_labels,
            k_values=k_values
        )
        
        # 추가 분석
        metrics.update(self._analyze_embedding_quality(
            query_embeddings, document_embeddings, queries, documents
        ))
        
        return metrics
    
    def evaluate_generation_model(
        self,
        model_path: str,
        eval_data_path: str,
        generation_config: Optional[str] = None,
        max_samples: Optional[int] = None,
        temperature: float = 0.7,
        max_tokens: int = 800
    ) -> Dict[str, Any]:
        """생성 모델 평가"""
        logger.info("생성 모델 평가 시작")
        
        # 모델 로드
        generator = FragranceGenerator()
        generator.load_model(model_path)
        
        # 생성 설정 로드
        gen_config = {"temperature": temperature, "max_tokens": max_tokens}
        if generation_config and os.path.exists(generation_config):
            with open(generation_config, 'r') as f:
                gen_config.update(json.load(f))
        
        # 평가 데이터 로드
        data_loader = DatasetLoader()
        eval_prompts = data_loader.load_generation_eval_dataset(eval_data_path)
        
        if max_samples:
            eval_prompts = eval_prompts[:max_samples]
        
        logger.info(f"평가 프롬프트 수: {len(eval_prompts)}")
        
        # 레시피 생성
        generated_recipes = []
        generation_times = []
        
        logger.info("레시피 생성 중...")
        for prompt in tqdm(eval_prompts):
            start_time = datetime.now()
            
            try:
                result = generator.generate(prompt, generation_config=gen_config)
                generated_recipes.append(result)
                
                end_time = datetime.now()
                generation_times.append((end_time - start_time).total_seconds())
                
            except Exception as e:
                logger.warning(f"생성 실패: {e}")
                generated_recipes.append({"error": str(e)})
                generation_times.append(0)
        
        # 메트릭 계산
        logger.info("메트릭 계산 중...")
        
        # 성공적으로 생성된 레시피만 필터링
        valid_recipes = [r for r in generated_recipes if "error" not in r]
        
        metrics = EvaluationMetrics.calculate_generation_metrics(valid_recipes)
        
        # 생성 성능 메트릭 추가
        metrics.update({
            "generation_success_rate": len(valid_recipes) / len(generated_recipes),
            "avg_generation_time": np.mean(generation_times),
            "generation_time_std": np.std(generation_times),
            "total_generated": len(generated_recipes),
            "valid_generated": len(valid_recipes)
        })
        
        # 개별 레시피 품질 평가
        quality_scores = []
        for recipe in tqdm(valid_recipes, desc="품질 평가"):
            quality_result = self.quality_assessor.assess_recipe_quality(recipe)
            quality_scores.append(quality_result["overall_score"])
        
        if quality_scores:
            metrics.update({
                "avg_quality_score": np.mean(quality_scores),
                "quality_score_std": np.std(quality_scores),
                "quality_score_min": np.min(quality_scores),
                "quality_score_max": np.max(quality_scores)
            })
        
        # 결과 저장
        if valid_recipes:
            results_path = os.path.join(self.output_dir, "generated_recipes.json")
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "recipes": valid_recipes,
                    "quality_scores": quality_scores,
                    "generation_times": generation_times,
                    "config": gen_config
                }, f, ensure_ascii=False, indent=2)
            
            logger.info(f"생성 결과 저장: {results_path}")
        
        return metrics
    
    def evaluate_search_system(
        self,
        model_path: str,
        eval_data_path: str,
        k_values: List[int] = [1, 5, 10, 20],
        max_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """검색 시스템 평가"""
        logger.info("검색 시스템 평가 시작")
        
        # 검색 서비스 초기화
        search_service = SearchService()
        # 여기서는 실제 초기화 대신 mock을 사용할 수 있음
        
        # 평가 데이터 로드
        data_loader = DatasetLoader()
        eval_queries = data_loader.load_search_eval_dataset(eval_data_path)
        
        if max_samples:
            eval_queries = eval_queries[:max_samples]
        
        logger.info(f"평가 쿼리 수: {len(eval_queries)}")
        
        # 검색 성능 평가
        search_results = []
        search_times = []
        relevance_scores = []
        
        logger.info("검색 수행 중...")
        for query_data in tqdm(eval_queries):
            query = query_data["query"]
            expected_results = query_data.get("expected_results", [])
            
            start_time = datetime.now()
            
            try:
                # 검색 수행
                result = search_service.semantic_search(
                    query=query,
                    top_k=max(k_values),
                    similarity_threshold=0.5
                )
                
                end_time = datetime.now()
                search_time = (end_time - start_time).total_seconds()
                
                search_results.append(result)
                search_times.append(search_time)
                
                # 관련성 점수 계산
                if expected_results:
                    relevance_score = self._calculate_search_relevance(
                        result["results"], expected_results
                    )
                    relevance_scores.append(relevance_score)
                
            except Exception as e:
                logger.warning(f"검색 실패: {e}")
                search_results.append({"error": str(e)})
                search_times.append(0)
                relevance_scores.append(0)
        
        # 메트릭 계산
        valid_results = [r for r in search_results if "error" not in r]
        
        metrics = {
            "search_success_rate": len(valid_results) / len(search_results),
            "avg_search_time": np.mean(search_times),
            "search_time_std": np.std(search_times),
            "total_searches": len(search_results),
            "valid_searches": len(valid_results)
        }
        
        if relevance_scores:
            metrics.update({
                "avg_relevance_score": np.mean(relevance_scores),
                "relevance_score_std": np.std(relevance_scores)
            })
        
        return metrics
    
    def evaluate_end_to_end(
        self,
        model_path: str,
        eval_data_path: str,
        max_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """End-to-end 시스템 평가"""
        logger.info("End-to-end 시스템 평가 시작")
        
        # 서비스 초기화
        search_service = SearchService()
        generation_service = GenerationService()
        
        # 평가 데이터 로드
        data_loader = DatasetLoader()
        eval_scenarios = data_loader.load_e2e_eval_dataset(eval_data_path)
        
        if max_samples:
            eval_scenarios = eval_scenarios[:max_samples]
        
        logger.info(f"평가 시나리오 수: {len(eval_scenarios)}")
        
        # End-to-end 평가
        e2e_results = []
        total_times = []
        success_count = 0
        
        logger.info("End-to-end 평가 수행 중...")
        for scenario in tqdm(eval_scenarios):
            start_time = datetime.now()
            
            try:
                # 1단계: 검색
                search_query = scenario["search_query"]
                search_result = search_service.semantic_search(
                    query=search_query,
                    top_k=5
                )
                
                # 2단계: 생성
                generation_prompt = self._create_generation_prompt(
                    scenario, search_result["results"]
                )
                generation_result = generation_service.generate_recipe(
                    generation_prompt
                )
                
                end_time = datetime.now()
                total_time = (end_time - start_time).total_seconds()
                
                e2e_results.append({
                    "scenario": scenario,
                    "search_result": search_result,
                    "generation_result": generation_result,
                    "total_time": total_time,
                    "success": True
                })
                
                total_times.append(total_time)
                success_count += 1
                
            except Exception as e:
                logger.warning(f"End-to-end 평가 실패: {e}")
                e2e_results.append({
                    "scenario": scenario,
                    "error": str(e),
                    "success": False
                })
                total_times.append(0)
        
        # 메트릭 계산
        metrics = {
            "e2e_success_rate": success_count / len(eval_scenarios),
            "avg_total_time": np.mean(total_times) if total_times else 0,
            "total_time_std": np.std(total_times) if total_times else 0,
            "total_scenarios": len(eval_scenarios),
            "successful_scenarios": success_count
        }
        
        # 결과 저장
        results_path = os.path.join(self.output_dir, "e2e_results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(e2e_results, f, ensure_ascii=False, indent=2, default=str)
        
        return metrics
    
    def _analyze_embedding_quality(
        self, 
        query_embeddings: np.ndarray,
        document_embeddings: np.ndarray,
        queries: List[str],
        documents: List[str]
    ) -> Dict[str, Any]:
        """임베딩 품질 분석"""
        # 임베딩 차원 분석
        query_dim_stats = {
            "mean": np.mean(np.linalg.norm(query_embeddings, axis=1)),
            "std": np.std(np.linalg.norm(query_embeddings, axis=1)),
            "min": np.min(np.linalg.norm(query_embeddings, axis=1)),
            "max": np.max(np.linalg.norm(query_embeddings, axis=1))
        }
        
        doc_dim_stats = {
            "mean": np.mean(np.linalg.norm(document_embeddings, axis=1)),
            "std": np.std(np.linalg.norm(document_embeddings, axis=1)),
            "min": np.min(np.linalg.norm(document_embeddings, axis=1)),
            "max": np.max(np.linalg.norm(document_embeddings, axis=1))
        }
        
        return {
            "query_embedding_stats": query_dim_stats,
            "document_embedding_stats": doc_dim_stats,
            "embedding_dimension": query_embeddings.shape[1]
        }
    
    def _calculate_search_relevance(
        self, 
        search_results: List[Dict[str, Any]], 
        expected_results: List[str]
    ) -> float:
        """검색 결과 관련성 점수 계산"""
        if not search_results or not expected_results:
            return 0.0
        
        result_ids = [r.get("id", "") for r in search_results]
        matches = sum(1 for expected_id in expected_results if expected_id in result_ids)
        
        return matches / len(expected_results)
    
    def _create_generation_prompt(
        self, 
        scenario: Dict[str, Any], 
        search_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """생성을 위한 프롬프트 생성"""
        # 검색 결과를 기반으로 생성 프롬프트 구성
        context = "\n".join([
            f"- {result.get('document', '')}" 
            for result in search_results[:3]
        ])
        
        return {
            "fragrance_family": scenario.get("fragrance_family", "floral"),
            "mood": scenario.get("mood", "romantic"),
            "intensity": scenario.get("intensity", "moderate"),
            "context": context,
            "generation_type": "basic_recipe"
        }

def main():
    """메인 함수"""
    args = parse_arguments()
    
    # 평가자 초기화
    evaluator = ModelEvaluator(args.output_dir)
    
    logger.info(f"모델 평가 시작 - 타입: {args.model_type}")
    logger.info(f"모델 경로: {args.model_path}")
    logger.info(f"평가 데이터: {args.eval_data}")
    
    try:
        # 모델 타입별 평가 실행
        if args.model_type == "embedding":
            metrics = evaluator.evaluate_embedding_model(
                model_path=args.model_path,
                eval_data_path=args.eval_data,
                batch_size=args.batch_size,
                max_samples=args.max_samples,
                k_values=args.k_values
            )
        
        elif args.model_type == "generation":
            metrics = evaluator.evaluate_generation_model(
                model_path=args.model_path,
                eval_data_path=args.eval_data,
                generation_config=args.generation_config,
                max_samples=args.max_samples,
                temperature=args.temperature,
                max_tokens=args.max_tokens
            )
        
        elif args.model_type == "search":
            metrics = evaluator.evaluate_search_system(
                model_path=args.model_path,
                eval_data_path=args.eval_data,
                k_values=args.k_values,
                max_samples=args.max_samples
            )
        
        elif args.model_type == "end-to-end":
            metrics = evaluator.evaluate_end_to_end(
                model_path=args.model_path,
                eval_data_path=args.eval_data,
                max_samples=args.max_samples
            )
        
        # 결과 출력
        logger.info("평가 결과:")
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
        
        # 결과 저장
        results_path = os.path.join(args.output_dir, "evaluation_metrics.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        
        # 요약 리포트 생성
        report_path = os.path.join(args.output_dir, "evaluation_report.md")
        evaluator._generate_evaluation_report(metrics, report_path, args)
        
        logger.info(f"평가 완료! 결과: {results_path}")
        logger.info(f"리포트: {report_path}")
        
    except Exception as e:
        logger.error(f"평가 중 오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()