from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class EvaluationMetrics:
    """모델 평가 메트릭 클래스"""
    
    @staticmethod
    def calculate_embedding_metrics(
        query_embeddings: np.ndarray,
        document_embeddings: np.ndarray,
        relevance_labels: np.ndarray,
        k_values: List[int] = [1, 5, 10, 20]
    ) -> Dict[str, float]:
        """
        임베딩 모델 평가 메트릭
        
        Args:
            query_embeddings: 쿼리 임베딩 (N, D)
            document_embeddings: 문서 임베딩 (M, D) 
            relevance_labels: 관련성 라벨 (N, M) - 1: 관련, 0: 비관련
            k_values: 평가할 k 값들
        """
        metrics = {}
        
        # 코사인 유사도 계산
        similarity_matrix = cosine_similarity(query_embeddings, document_embeddings)
        
        # 각 k 값에 대해 메트릭 계산
        for k in k_values:
            # Top-k 문서 인덱스 추출
            top_k_indices = np.argsort(similarity_matrix, axis=1)[:, -k:][:, ::-1]
            
            # Precision@K 계산
            precision_k = EvaluationMetrics._calculate_precision_at_k(
                top_k_indices, relevance_labels, k
            )
            
            # Recall@K 계산
            recall_k = EvaluationMetrics._calculate_recall_at_k(
                top_k_indices, relevance_labels, k
            )
            
            # NDCG@K 계산
            ndcg_k = EvaluationMetrics._calculate_ndcg_at_k(
                similarity_matrix, relevance_labels, k
            )
            
            # MAP@K 계산
            map_k = EvaluationMetrics._calculate_map_at_k(
                similarity_matrix, relevance_labels, k
            )
            
            metrics[f'precision@{k}'] = precision_k
            metrics[f'recall@{k}'] = recall_k
            metrics[f'ndcg@{k}'] = ndcg_k
            metrics[f'map@{k}'] = map_k
        
        # MRR (Mean Reciprocal Rank) 계산
        mrr = EvaluationMetrics._calculate_mrr(similarity_matrix, relevance_labels)
        metrics['mrr'] = mrr
        
        # 평균 유사도 점수
        avg_similarity = np.mean([
            np.max(similarity_matrix[i, relevance_labels[i] == 1])
            for i in range(len(query_embeddings))
            if np.any(relevance_labels[i] == 1)
        ])
        metrics['avg_similarity'] = avg_similarity
        
        return metrics
    
    @staticmethod
    def _calculate_precision_at_k(
        top_k_indices: np.ndarray, 
        relevance_labels: np.ndarray, 
        k: int
    ) -> float:
        """Precision@K 계산"""
        precisions = []
        for i, indices in enumerate(top_k_indices):
            relevant_retrieved = np.sum(relevance_labels[i, indices[:k]])
            precision = relevant_retrieved / min(k, len(indices))
            precisions.append(precision)
        return np.mean(precisions)
    
    @staticmethod
    def _calculate_recall_at_k(
        top_k_indices: np.ndarray, 
        relevance_labels: np.ndarray, 
        k: int
    ) -> float:
        """Recall@K 계산"""
        recalls = []
        for i, indices in enumerate(top_k_indices):
            relevant_total = np.sum(relevance_labels[i])
            if relevant_total == 0:
                continue
            relevant_retrieved = np.sum(relevance_labels[i, indices[:k]])
            recall = relevant_retrieved / relevant_total
            recalls.append(recall)
        return np.mean(recalls) if recalls else 0.0
    
    @staticmethod
    def _calculate_ndcg_at_k(
        similarity_matrix: np.ndarray, 
        relevance_labels: np.ndarray, 
        k: int
    ) -> float:
        """NDCG@K 계산"""
        ndcg_scores = []
        
        for i in range(len(similarity_matrix)):
            # 유사도 기준으로 정렬된 인덱스
            sorted_indices = np.argsort(similarity_matrix[i])[::-1]
            
            # DCG 계산
            dcg = 0.0
            for j, doc_idx in enumerate(sorted_indices[:k]):
                if j == 0:
                    dcg += relevance_labels[i, doc_idx]
                else:
                    dcg += relevance_labels[i, doc_idx] / np.log2(j + 1)
            
            # IDCG 계산 (이상적인 DCG)
            ideal_relevance = np.sort(relevance_labels[i])[::-1]
            idcg = 0.0
            for j, rel in enumerate(ideal_relevance[:k]):
                if j == 0:
                    idcg += rel
                else:
                    idcg += rel / np.log2(j + 1)
            
            # NDCG 계산
            if idcg > 0:
                ndcg_scores.append(dcg / idcg)
            else:
                ndcg_scores.append(0.0)
        
        return np.mean(ndcg_scores)
    
    @staticmethod
    def _calculate_map_at_k(
        similarity_matrix: np.ndarray, 
        relevance_labels: np.ndarray, 
        k: int
    ) -> float:
        """MAP@K 계산"""
        ap_scores = []
        
        for i in range(len(similarity_matrix)):
            sorted_indices = np.argsort(similarity_matrix[i])[::-1]
            
            relevant_docs = 0
            precision_sum = 0.0
            
            for j, doc_idx in enumerate(sorted_indices[:k]):
                if relevance_labels[i, doc_idx] == 1:
                    relevant_docs += 1
                    precision_sum += relevant_docs / (j + 1)
            
            total_relevant = np.sum(relevance_labels[i])
            if total_relevant > 0:
                ap_scores.append(precision_sum / min(total_relevant, k))
            else:
                ap_scores.append(0.0)
        
        return np.mean(ap_scores)
    
    @staticmethod
    def _calculate_mrr(
        similarity_matrix: np.ndarray, 
        relevance_labels: np.ndarray
    ) -> float:
        """MRR (Mean Reciprocal Rank) 계산"""
        reciprocal_ranks = []
        
        for i in range(len(similarity_matrix)):
            sorted_indices = np.argsort(similarity_matrix[i])[::-1]
            
            for rank, doc_idx in enumerate(sorted_indices):
                if relevance_labels[i, doc_idx] == 1:
                    reciprocal_ranks.append(1.0 / (rank + 1))
                    break
            else:
                reciprocal_ranks.append(0.0)
        
        return np.mean(reciprocal_ranks)
    
    @staticmethod
    def calculate_generation_metrics(
        generated_recipes: List[Dict[str, Any]],
        reference_recipes: Optional[List[Dict[str, Any]]] = None,
        evaluation_criteria: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        생성 모델 평가 메트릭
        
        Args:
            generated_recipes: 생성된 레시피들
            reference_recipes: 참조 레시피들 (옵션)
            evaluation_criteria: 평가 기준 (옵션)
        """
        metrics = {}
        
        if not generated_recipes:
            return {"error": "No generated recipes provided"}
        
        # 구조 완성도 평가
        structure_scores = []
        for recipe in generated_recipes:
            score = EvaluationMetrics._evaluate_recipe_structure(recipe)
            structure_scores.append(score)
        
        metrics['avg_structure_score'] = np.mean(structure_scores)
        metrics['structure_std'] = np.std(structure_scores)
        
        # 창의성 평가
        creativity_scores = []
        for recipe in generated_recipes:
            score = EvaluationMetrics._evaluate_recipe_creativity(recipe)
            creativity_scores.append(score)
        
        metrics['avg_creativity_score'] = np.mean(creativity_scores)
        metrics['creativity_std'] = np.std(creativity_scores)
        
        # 실현 가능성 평가
        feasibility_scores = []
        for recipe in generated_recipes:
            score = EvaluationMetrics._evaluate_recipe_feasibility(recipe)
            feasibility_scores.append(score)
        
        metrics['avg_feasibility_score'] = np.mean(feasibility_scores)
        metrics['feasibility_std'] = np.std(feasibility_scores)
        
        # 일관성 평가
        consistency_scores = []
        for recipe in generated_recipes:
            score = EvaluationMetrics._evaluate_recipe_consistency(recipe)
            consistency_scores.append(score)
        
        metrics['avg_consistency_score'] = np.mean(consistency_scores)
        metrics['consistency_std'] = np.std(consistency_scores)
        
        # 전체 품질 점수
        quality_scores = []
        for i, recipe in enumerate(generated_recipes):
            quality_score = (
                structure_scores[i] * 0.3 +
                creativity_scores[i] * 0.25 +
                feasibility_scores[i] * 0.25 +
                consistency_scores[i] * 0.2
            )
            quality_scores.append(quality_score)
        
        metrics['avg_quality_score'] = np.mean(quality_scores)
        metrics['quality_std'] = np.std(quality_scores)
        
        # 다양성 평가
        diversity_score = EvaluationMetrics._calculate_recipe_diversity(generated_recipes)
        metrics['diversity_score'] = diversity_score
        
        # 참조 레시피와의 비교 (있는 경우)
        if reference_recipes:
            similarity_scores = EvaluationMetrics._compare_with_reference(
                generated_recipes, reference_recipes
            )
            metrics.update(similarity_scores)
        
        return metrics
    
    @staticmethod
    def _evaluate_recipe_structure(recipe: Dict[str, Any]) -> float:
        """레시피 구조 평가"""
        required_fields = ["name", "description", "notes", "formula"]
        score = 0.0
        
        for field in required_fields:
            if field in recipe and recipe[field]:
                score += 0.25
        
        # 노트 구조 확인
        if "notes" in recipe:
            note_types = ["top", "middle", "base"]
            for note_type in note_types:
                if note_type in recipe["notes"] and recipe["notes"][note_type]:
                    score += 0.1
        
        return min(1.0, score)
    
    @staticmethod
    def _evaluate_recipe_creativity(recipe: Dict[str, Any]) -> float:
        """레시피 창의성 평가"""
        creativity_score = 0.0
        
        # 노트 다양성
        if "notes" in recipe:
            all_notes = []
            for note_list in recipe["notes"].values():
                if isinstance(note_list, list):
                    all_notes.extend([
                        note.get("name", note) if isinstance(note, dict) else note
                        for note in note_list
                    ])
            
            unique_ratio = len(set(all_notes)) / max(len(all_notes), 1)
            creativity_score += unique_ratio * 0.3
        
        # 설명의 풍부함
        description = recipe.get("description", "")
        if len(description) > 50:
            creativity_score += 0.2
        
        # 특별한 컨셉트 존재
        concept_fields = ["concept", "artisan_concept", "heritage_story"]
        if any(field in recipe for field in concept_fields):
            creativity_score += 0.3
        
        # 독특한 노트 조합
        if "signature_notes" in recipe or "exclusive_formula" in recipe:
            creativity_score += 0.2
        
        return min(1.0, creativity_score)
    
    @staticmethod
    def _evaluate_recipe_feasibility(recipe: Dict[str, Any]) -> float:
        """레시피 실현 가능성 평가"""
        feasibility_score = 1.0
        
        # 농도 합계 확인
        if "formula" in recipe and isinstance(recipe["formula"], dict):
            total_percentage = 0
            for ingredient, percentage in recipe["formula"].items():
                if isinstance(percentage, (int, float)):
                    total_percentage += percentage
                elif isinstance(percentage, str):
                    try:
                        total_percentage += float(percentage.replace('%', ''))
                    except:
                        continue
            
            if total_percentage > 120:
                feasibility_score -= 0.3
            elif total_percentage < 50:
                feasibility_score -= 0.2
        
        return max(0.0, feasibility_score)
    
    @staticmethod
    def _evaluate_recipe_consistency(recipe: Dict[str, Any]) -> float:
        """레시피 일관성 평가"""
        consistency_score = 1.0
        
        if "notes" not in recipe or "fragrance_family" not in recipe:
            return 0.3
        
        notes = recipe["notes"]
        family = recipe["fragrance_family"].lower()
        
        # 향조 패밀리와 노트의 일관성 확인
        family_note_mapping = {
            "citrus": {
                "expected": ["lemon", "orange", "bergamot", "grapefruit", "lime", "mandarin"],
                "unexpected": ["vanilla", "amber", "musk", "sandalwood"]
            },
            "floral": {
                "expected": ["rose", "jasmine", "lily", "peony", "iris", "violet", "gardenia"],
                "unexpected": ["tobacco", "leather", "cedar", "pine"]
            },
            "woody": {
                "expected": ["sandalwood", "cedar", "oak", "birch", "rosewood", "teak"],
                "unexpected": ["lemon", "orange", "raspberry", "strawberry"]
            },
            "oriental": {
                "expected": ["vanilla", "amber", "incense", "patchouli", "oud", "benzoin"],
                "unexpected": ["apple", "pear", "cucumber", "watermelon"]
            },
            "fresh": {
                "expected": ["mint", "eucalyptus", "marine", "ozone", "cucumber", "green leaves"],
                "unexpected": ["vanilla", "amber", "chocolate", "coffee"]
            }
        }
        
        # 모든 노트 수집
        all_notes = []
        for note_list in notes.values():
            if isinstance(note_list, list):
                for note in note_list:
                    note_name = note.get("name", note) if isinstance(note, dict) else note
                    all_notes.append(note_name.lower())
        
        if family in family_note_mapping:
            expected_notes = family_note_mapping[family]["expected"]
            unexpected_notes = family_note_mapping[family]["unexpected"]
            
            # 기대되는 노트가 있는지 확인
            expected_found = sum(1 for note in all_notes if any(exp in note for exp in expected_notes))
            unexpected_found = sum(1 for note in all_notes if any(unexp in note for unexp in unexpected_notes))
            
            # 일관성 점수 조정
            if expected_found == 0:
                consistency_score -= 0.3  # 기대되는 노트가 전혀 없음
            
            if unexpected_found > 0:
                consistency_score -= 0.2 * min(unexpected_found / len(all_notes), 0.5)  # 예상치 못한 노트
        
        # 노트 간의 화학적 조화 확인
        conflicting_pairs = [
            (["citrus", "lemon", "orange"], ["vanilla", "amber"]),  # 시트러스와 오리엔탈의 강한 대비
            (["marine", "ozone", "cucumber"], ["tobacco", "leather"]),  # 프레시와 스모키의 대비
            (["rose", "jasmine"], ["pine", "cedar"])  # 플로럴과 우디의 극단적 대비
        ]
        
        for group1, group2 in conflicting_pairs:
            has_group1 = any(note for note in all_notes if any(g1 in note for g1 in group1))
            has_group2 = any(note for note in all_notes if any(g2 in note for g2 in group2))
            
            if has_group1 and has_group2:
                consistency_score -= 0.1
        
        # 농도 일관성 확인 (포뮬러가 있는 경우)
        if "formula" in recipe and isinstance(recipe["formula"], dict):
            concentrations = []
            for ingredient, concentration in recipe["formula"].items():
                try:
                    if isinstance(concentration, (int, float)):
                        concentrations.append(concentration)
                    elif isinstance(concentration, str):
                        conc_val = float(concentration.replace('%', ''))
                        concentrations.append(conc_val)
                except:
                    continue
            
            if concentrations:
                # 농도 분포가 너무 불균등하면 일관성 저하
                if len(concentrations) > 1:
                    max_conc = max(concentrations)
                    min_conc = min(concentrations)
                    if max_conc > 0 and (max_conc / min_conc) > 20:  # 20배 이상 차이
                        consistency_score -= 0.1
        
        return max(0.0, consistency_score)
    
    @staticmethod
    def _calculate_recipe_diversity(recipes: List[Dict[str, Any]]) -> float:
        """레시피 다양성 계산"""
        if len(recipes) < 2:
            return 1.0
        
        # 향조 다양성
        families = [recipe.get("fragrance_family", "") for recipe in recipes]
        family_diversity = len(set(families)) / len(families)
        
        # 노트 다양성
        all_notes = set()
        total_notes = 0
        
        for recipe in recipes:
            if "notes" in recipe:
                for note_list in recipe["notes"].values():
                    if isinstance(note_list, list):
                        for note in note_list:
                            note_name = note.get("name", note) if isinstance(note, dict) else note
                            all_notes.add(note_name)
                            total_notes += 1
        
        note_diversity = len(all_notes) / max(total_notes, 1)
        
        return (family_diversity + note_diversity) / 2
    
    @staticmethod
    def _compare_with_reference(
        generated_recipes: List[Dict[str, Any]], 
        reference_recipes: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """참조 레시피와의 비교"""
        if not generated_recipes or not reference_recipes:
            return {"reference_similarity": 0.0, "reference_coverage": 0.0}
        
        similarities = []
        covered_elements = set()
        total_reference_elements = set()
        
        # 참조 레시피의 모든 요소 수집
        for ref_recipe in reference_recipes:
            # 향조 패밀리
            if "fragrance_family" in ref_recipe:
                total_reference_elements.add(f"family:{ref_recipe['fragrance_family']}")
            
            # 노트들
            if "notes" in ref_recipe:
                for note_type, notes in ref_recipe["notes"].items():
                    if isinstance(notes, list):
                        for note in notes:
                            note_name = note.get("name", note) if isinstance(note, dict) else note
                            total_reference_elements.add(f"note:{note_name}")
        
        # 생성된 레시피와 참조 레시피 비교
        for gen_recipe in generated_recipes:
            best_similarity = 0.0
            recipe_covered_elements = set()
            
            for ref_recipe in reference_recipes:
                similarity = EvaluationMetrics._calculate_recipe_similarity(gen_recipe, ref_recipe)
                best_similarity = max(best_similarity, similarity)
                
                # 커버된 요소들 추적
                if "fragrance_family" in gen_recipe and gen_recipe["fragrance_family"] == ref_recipe.get("fragrance_family"):
                    recipe_covered_elements.add(f"family:{gen_recipe['fragrance_family']}")
                
                # 노트 매칭
                if "notes" in gen_recipe and "notes" in ref_recipe:
                    for note_type in ["top", "middle", "base"]:
                        gen_notes = gen_recipe["notes"].get(note_type, [])
                        ref_notes = ref_recipe["notes"].get(note_type, [])
                        
                        for gen_note in gen_notes:
                            gen_note_name = gen_note.get("name", gen_note) if isinstance(gen_note, dict) else gen_note
                            for ref_note in ref_notes:
                                ref_note_name = ref_note.get("name", ref_note) if isinstance(ref_note, dict) else ref_note
                                if gen_note_name.lower() in ref_note_name.lower() or ref_note_name.lower() in gen_note_name.lower():
                                    recipe_covered_elements.add(f"note:{ref_note_name}")
            
            similarities.append(best_similarity)
            covered_elements.update(recipe_covered_elements)
        
        # 평균 유사도
        avg_similarity = np.mean(similarities) if similarities else 0.0
        
        # 커버리지 계산
        coverage = len(covered_elements) / len(total_reference_elements) if total_reference_elements else 0.0
        
        return {
            "reference_similarity": float(avg_similarity),
            "reference_coverage": float(coverage)
        }
    
    @staticmethod
    def _calculate_recipe_similarity(recipe1: Dict[str, Any], recipe2: Dict[str, Any]) -> float:
        """두 레시피 간의 유사도 계산"""
        similarity_score = 0.0
        total_weight = 0.0
        
        # 향조 패밀리 비교 (가중치: 0.3)
        if "fragrance_family" in recipe1 and "fragrance_family" in recipe2:
            if recipe1["fragrance_family"] == recipe2["fragrance_family"]:
                similarity_score += 0.3
            total_weight += 0.3
        
        # 노트 유사도 비교 (가중치: 0.6)
        if "notes" in recipe1 and "notes" in recipe2:
            note_similarity = EvaluationMetrics._calculate_note_similarity(
                recipe1["notes"], recipe2["notes"]
            )
            similarity_score += note_similarity * 0.6
            total_weight += 0.6
        
        # 설명 유사도 (가중치: 0.1)
        if "description" in recipe1 and "description" in recipe2:
            desc_similarity = EvaluationMetrics._calculate_text_similarity(
                recipe1["description"], recipe2["description"]
            )
            similarity_score += desc_similarity * 0.1
            total_weight += 0.1
        
        return similarity_score / total_weight if total_weight > 0 else 0.0
    
    @staticmethod
    def _calculate_note_similarity(notes1: Dict[str, Any], notes2: Dict[str, Any]) -> float:
        """노트 유사도 계산"""
        all_similarities = []
        
        for note_type in ["top", "middle", "base"]:
            notes_a = notes1.get(note_type, [])
            notes_b = notes2.get(note_type, [])
            
            if not notes_a or not notes_b:
                continue
            
            # 노트 이름 추출
            names_a = set()
            names_b = set()
            
            for note in notes_a:
                name = note.get("name", note) if isinstance(note, dict) else note
                names_a.add(name.lower())
            
            for note in notes_b:
                name = note.get("name", note) if isinstance(note, dict) else note
                names_b.add(name.lower())
            
            # Jaccard 유사도
            if names_a or names_b:
                intersection = names_a.intersection(names_b)
                union = names_a.union(names_b)
                similarity = len(intersection) / len(union) if union else 0.0
                all_similarities.append(similarity)
        
        return np.mean(all_similarities) if all_similarities else 0.0
    
    @staticmethod
    def _calculate_text_similarity(text1: str, text2: str) -> float:
        """텍스트 유사도 계산 (간단한 버전)"""
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0

class QualityAssessment:
    """레시피 품질 평가 클래스"""
    
    def __init__(self):
        self.quality_weights = {
            "structure": 0.25,
            "balance": 0.25,
            "creativity": 0.2,
            "feasibility": 0.15,
            "consistency": 0.15
        }
    
    def assess_recipe_quality(
        self, 
        recipe: Dict[str, Any],
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """레시피 품질 종합 평가"""
        if weights:
            self.quality_weights.update(weights)
        
        assessments = {}
        
        # 구조 평가
        structure_score = self._assess_structure(recipe)
        assessments["structure"] = {
            "score": structure_score,
            "weight": self.quality_weights["structure"],
            "details": self._get_structure_details(recipe)
        }
        
        # 균형성 평가
        balance_score = self._assess_balance(recipe)
        assessments["balance"] = {
            "score": balance_score,
            "weight": self.quality_weights["balance"],
            "details": self._get_balance_details(recipe)
        }
        
        # 창의성 평가
        creativity_score = self._assess_creativity(recipe)
        assessments["creativity"] = {
            "score": creativity_score,
            "weight": self.quality_weights["creativity"],
            "details": self._get_creativity_details(recipe)
        }
        
        # 실현가능성 평가
        feasibility_score = self._assess_feasibility(recipe)
        assessments["feasibility"] = {
            "score": feasibility_score,
            "weight": self.quality_weights["feasibility"],
            "details": self._get_feasibility_details(recipe)
        }
        
        # 일관성 평가
        consistency_score = self._assess_consistency(recipe)
        assessments["consistency"] = {
            "score": consistency_score,
            "weight": self.quality_weights["consistency"],
            "details": self._get_consistency_details(recipe)
        }
        
        # 종합 점수 계산
        total_score = sum(
            assessment["score"] * assessment["weight"]
            for assessment in assessments.values()
        )
        
        # 등급 결정
        grade = self._determine_grade(total_score)
        
        return {
            "overall_score": round(total_score * 100, 1),
            "grade": grade,
            "assessments": assessments,
            "recommendations": self._generate_recommendations(assessments),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _assess_structure(self, recipe: Dict[str, Any]) -> float:
        """구조 평가"""
        return EvaluationMetrics._evaluate_recipe_structure(recipe)
    
    def _assess_balance(self, recipe: Dict[str, Any]) -> float:
        """균형성 평가"""
        if "notes" not in recipe:
            return 0.3
        
        notes = recipe["notes"]
        top_count = len(notes.get("top", []))
        middle_count = len(notes.get("middle", []))
        base_count = len(notes.get("base", []))
        
        # 이상적인 비율 확인
        total = top_count + middle_count + base_count
        if total == 0:
            return 0.0
        
        top_ratio = top_count / total
        middle_ratio = middle_count / total
        base_ratio = base_count / total
        
        # 이상적인 비율: 톱 30-40%, 미들 40-50%, 베이스 20-30%
        score = 1.0
        
        if not (0.25 <= top_ratio <= 0.45):
            score -= 0.2
        if not (0.35 <= middle_ratio <= 0.55):
            score -= 0.3
        if not (0.15 <= base_ratio <= 0.35):
            score -= 0.2
        
        return max(0.0, score)
    
    def _assess_creativity(self, recipe: Dict[str, Any]) -> float:
        """창의성 평가"""
        return EvaluationMetrics._evaluate_recipe_creativity(recipe)
    
    def _assess_feasibility(self, recipe: Dict[str, Any]) -> float:
        """실현가능성 평가"""
        return EvaluationMetrics._evaluate_recipe_feasibility(recipe)
    
    def _assess_consistency(self, recipe: Dict[str, Any]) -> float:
        """일관성 평가"""
        return EvaluationMetrics._evaluate_recipe_consistency(recipe)
    
    def _get_structure_details(self, recipe: Dict[str, Any]) -> Dict[str, Any]:
        """구조 평가 상세"""
        required_fields = ["name", "description", "notes", "formula"]
        present_fields = [field for field in required_fields if field in recipe and recipe[field]]
        
        return {
            "required_fields": required_fields,
            "present_fields": present_fields,
            "completion_rate": len(present_fields) / len(required_fields)
        }
    
    def _get_balance_details(self, recipe: Dict[str, Any]) -> Dict[str, Any]:
        """균형성 평가 상세"""
        if "notes" not in recipe:
            return {"error": "No notes found"}
        
        notes = recipe["notes"]
        counts = {
            "top": len(notes.get("top", [])),
            "middle": len(notes.get("middle", [])),
            "base": len(notes.get("base", []))
        }
        
        total = sum(counts.values())
        ratios = {k: v/total if total > 0 else 0 for k, v in counts.items()}
        
        return {
            "note_counts": counts,
            "note_ratios": ratios,
            "total_notes": total
        }
    
    def _get_creativity_details(self, recipe: Dict[str, Any]) -> Dict[str, Any]:
        """창의성 평가 상세"""
        details = {
            "has_concept": any(field in recipe for field in ["concept", "artisan_concept"]),
            "description_length": len(recipe.get("description", "")),
            "unique_elements": []
        }
        
        if "signature_notes" in recipe:
            details["unique_elements"].append("signature_notes")
        if "exclusive_formula" in recipe:
            details["unique_elements"].append("exclusive_formula")
        
        return details
    
    def _get_feasibility_details(self, recipe: Dict[str, Any]) -> Dict[str, Any]:
        """실현가능성 평가 상세"""
        details = {"issues": [], "total_concentration": 0}
        
        if "formula" in recipe and isinstance(recipe["formula"], dict):
            total = 0
            for ingredient, percentage in recipe["formula"].items():
                if isinstance(percentage, (int, float)):
                    total += percentage
            
            details["total_concentration"] = total
            
            if total > 120:
                details["issues"].append("Total concentration exceeds 120%")
            elif total < 50:
                details["issues"].append("Total concentration below 50%")
        
        return details
    
    def _get_consistency_details(self, recipe: Dict[str, Any]) -> Dict[str, Any]:
        """일관성 평가 상세"""
        return {"note": "Basic consistency check performed"}
    
    def _determine_grade(self, score: float) -> str:
        """점수 기반 등급 결정"""
        if score >= 0.9:
            return "A+"
        elif score >= 0.85:
            return "A"
        elif score >= 0.8:
            return "A-"
        elif score >= 0.75:
            return "B+"
        elif score >= 0.7:
            return "B"
        elif score >= 0.65:
            return "B-"
        elif score >= 0.6:
            return "C+"
        elif score >= 0.55:
            return "C"
        elif score >= 0.5:
            return "C-"
        else:
            return "D"
    
    def _generate_recommendations(self, assessments: Dict[str, Any]) -> List[str]:
        """개선 권장사항 생성"""
        recommendations = []
        
        for aspect, data in assessments.items():
            if data["score"] < 0.7:
                if aspect == "structure":
                    recommendations.append("레시피 구조를 완성하세요 (이름, 설명, 노트, 포뮬러)")
                elif aspect == "balance":
                    recommendations.append("노트 비율의 균형을 맞춰보세요")
                elif aspect == "creativity":
                    recommendations.append("더 독창적인 요소를 추가해보세요")
                elif aspect == "feasibility":
                    recommendations.append("농도와 실현 가능성을 검토하세요")
                elif aspect == "consistency":
                    recommendations.append("레시피 전체의 일관성을 확인하세요")
        
        if not recommendations:
            recommendations.append("훌륭한 레시피입니다!")
        
        return recommendations