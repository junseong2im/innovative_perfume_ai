"""
Living Scent Orchestrator
모든 Living Scent AI 에이전트들을 조율하는 중앙 지휘자

사용자의 입력을 받아 적절한 AI 에이전트를 호출하고,
DNA 생성부터 진화까지의 전체 프로세스를 관리합니다.
"""

import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import asdict
from datetime import datetime
from sqlalchemy.orm import Session

# Living Scent AI 에이전트들
from fragrance_ai.models.living_scent.linguistic_receptor import (
    get_linguistic_receptor,
    UserIntent
)
from fragrance_ai.models.living_scent.cognitive_core import get_cognitive_core
from fragrance_ai.models.living_scent.olfactory_recombinator import get_olfactory_recombinator
from fragrance_ai.models.living_scent.epigenetic_variation import get_epigenetic_variation

# 데이터베이스 모델
from fragrance_ai.database.living_scent_models import (
    OlfactoryDNAModel,
    ScentPhenotypeModel,
    UserInteractionModel,
    FragranceEvolutionTreeModel
)

logger = logging.getLogger(__name__)


class LivingScentOrchestrator:
    """
    Living Scent 시스템의 중앙 오케스트레이터
    생명체의 탄생과 진화 과정을 조율
    """

    def __init__(self, db_session: Optional[Session] = None):
        # AI 에이전트 초기화
        self.linguistic_receptor = get_linguistic_receptor()
        self.cognitive_core = get_cognitive_core()
        self.olfactory_recombinator = get_olfactory_recombinator()
        self.epigenetic_variation = get_epigenetic_variation()

        # 데이터베이스 세션
        self.db = db_session

        # 메모리 캐시 (DB 없이도 작동)
        self.memory_dna_library = {}
        self.memory_phenotype_library = {}

        logger.info("Living Scent Orchestrator initialized")

    def _save_dna_to_db(self, dna: Any, user_id: Optional[str] = None) -> bool:
        """DNA를 데이터베이스에 저장"""
        if not self.db:
            # DB 없으면 메모리 캐시만 사용
            self.memory_dna_library[dna.dna_id] = dna
            return True

        try:
            # DNA 모델 생성
            dna_model = OlfactoryDNAModel(
                dna_id=dna.dna_id,
                lineage=dna.lineage,
                genotype=self._genotype_to_dict(dna.genotype),
                phenotype_potential=dna.phenotype_potential,
                story=dna.story,
                generation=dna.generation,
                fitness_score=dna.fitness_score,
                mutation_history=dna.mutation_history,
                created_by_user_id=user_id
            )

            self.db.add(dna_model)
            self.db.commit()

            # 메모리 캐시에도 저장
            self.memory_dna_library[dna.dna_id] = dna

            logger.info(f"DNA {dna.dna_id} saved to database")
            return True

        except Exception as e:
            logger.error(f"Failed to save DNA to database: {e}")
            self.db.rollback()
            # 실패해도 메모리 캐시에는 저장
            self.memory_dna_library[dna.dna_id] = dna
            return False

    def _save_phenotype_to_db(self, phenotype: Any, user_id: Optional[str] = None) -> bool:
        """표현형을 데이터베이스에 저장"""
        if not self.db:
            # DB 없으면 메모리 캐시만 사용
            self.memory_phenotype_library[phenotype.phenotype_id] = phenotype
            return True

        try:
            # 표현형 모델 생성
            phenotype_model = ScentPhenotypeModel(
                phenotype_id=phenotype.phenotype_id,
                based_on_dna=phenotype.based_on_dna,
                epigenetic_trigger=phenotype.epigenetic_trigger,
                recipe=phenotype.recipe,
                modifications=self._modifications_to_dict(phenotype.modifications),
                description=phenotype.description,
                environmental_response=phenotype.environmental_response,
                parent_phenotype=phenotype.parent_phenotype,
                evolution_path=phenotype.evolution_path,
                created_by_user_id=user_id
            )

            self.db.add(phenotype_model)
            self.db.commit()

            # 메모리 캐시에도 저장
            self.memory_phenotype_library[phenotype.phenotype_id] = phenotype

            logger.info(f"Phenotype {phenotype.phenotype_id} saved to database")
            return True

        except Exception as e:
            logger.error(f"Failed to save phenotype to database: {e}")
            self.db.rollback()
            # 실패해도 메모리 캐시에는 저장
            self.memory_phenotype_library[phenotype.phenotype_id] = phenotype
            return False

    def _genotype_to_dict(self, genotype: Dict) -> Dict:
        """유전자형을 딕셔너리로 변환"""
        result = {}
        for note_type, genes in genotype.items():
            result[note_type] = [
                {
                    'note_type': g.note_type,
                    'ingredient': g.ingredient,
                    'concentration': g.concentration,
                    'volatility': g.volatility,
                    'molecular_weight': g.molecular_weight,
                    'odor_family': g.odor_family,
                    'expression_level': g.expression_level
                }
                for g in genes
            ]
        return result

    def _modifications_to_dict(self, modifications: List) -> List[Dict]:
        """수정 기록을 딕셔너리로 변환"""
        return [
            {
                'marker_type': mod.marker_type.value,
                'target_gene': mod.target_gene,
                'modification_factor': mod.modification_factor,
                'trigger': mod.trigger,
                'timestamp': mod.timestamp
            }
            for mod in modifications
        ]

    def _save_interaction(
        self,
        user_id: str,
        interaction_type: str,
        dna_id: Optional[str] = None,
        phenotype_id: Optional[str] = None,
        interaction_data: Optional[Dict] = None
    ):
        """사용자 상호작용 저장"""
        if not self.db:
            return

        try:
            interaction = UserInteractionModel(
                user_id=user_id,
                interaction_type=interaction_type,
                dna_id=dna_id,
                phenotype_id=phenotype_id,
                interaction_data=interaction_data
            )
            self.db.add(interaction)
            self.db.commit()
        except Exception as e:
            logger.error(f"Failed to save interaction: {e}")
            self.db.rollback()

    def process_user_input(
        self,
        user_input: str,
        user_id: Optional[str] = None,
        existing_dna_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        사용자 입력을 처리하는 메인 함수
        전체 Living Scent 프로세스를 조율
        """
        logger.info(f"Processing user input: {user_input[:100]}...")

        try:
            # 1단계: 언어 수용체 - 텍스트 분석
            structured_input = self.linguistic_receptor.process(user_input)
            logger.info(f"Intent classified: {structured_input.intent}")

            # 2단계: 인지 코어 - 감정 해석
            creative_brief = self.cognitive_core.synthesize(structured_input)
            logger.info(f"Creative brief generated: {creative_brief.theme}")

            # 3단계: 의도에 따른 처리 분기
            if structured_input.intent == UserIntent.CREATE_NEW:
                # 새로운 DNA 생성
                result = self._create_new_dna(creative_brief, user_id)
                interaction_type = "create"

            elif structured_input.intent == UserIntent.EVOLVE_EXISTING:
                # 기존 DNA 진화
                if not existing_dna_id:
                    # DNA가 지정되지 않았으면 가장 최근 것 사용
                    if self.memory_dna_library:
                        existing_dna_id = list(self.memory_dna_library.keys())[-1]
                    else:
                        # 진화할 DNA가 없으면 새로 생성
                        logger.warning("No existing DNA found, creating new one")
                        result = self._create_new_dna(creative_brief, user_id)
                        interaction_type = "create"
                else:
                    result = self._evolve_existing_dna(
                        existing_dna_id,
                        creative_brief,
                        user_id
                    )
                    interaction_type = "evolve"
            else:
                # Unknown intent - 기본적으로 새로 생성
                result = self._create_new_dna(creative_brief, user_id)
                interaction_type = "create"

            # 4단계: 상호작용 기록
            if user_id:
                self._save_interaction(
                    user_id=user_id,
                    interaction_type=interaction_type,
                    dna_id=result.get('dna_id'),
                    phenotype_id=result.get('phenotype_id'),
                    interaction_data={
                        'user_input': user_input,
                        'intent': structured_input.intent.value,
                        'keywords': structured_input.keywords
                    }
                )

            # 5단계: 결과 포맷팅
            response = {
                'success': True,
                'intent': structured_input.intent.value,
                'result': result,
                'metadata': {
                    'theme': creative_brief.theme,
                    'core_emotion': creative_brief.core_emotion,
                    'story': creative_brief.story,
                    'keywords': structured_input.keywords,
                    'confidence': structured_input.confidence_score
                }
            }

            logger.info(f"Successfully processed user input")
            return response

        except Exception as e:
            logger.error(f"Error processing user input: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to process your request'
            }

    def _create_new_dna(self, creative_brief: Any, user_id: Optional[str] = None) -> Dict[str, Any]:
        """새로운 DNA 생성"""
        # DNA 생성
        olfactory_dna = self.olfactory_recombinator.create(creative_brief)

        # 데이터베이스 저장
        self._save_dna_to_db(olfactory_dna, user_id)

        # 레시피 변환
        recipe = self.olfactory_recombinator.to_recipe(olfactory_dna)

        return {
            'type': 'new_dna',
            'dna_id': olfactory_dna.dna_id,
            'lineage': olfactory_dna.lineage,
            'recipe': recipe,
            'story': olfactory_dna.story,
            'phenotype_potential': olfactory_dna.phenotype_potential,
            'generation': olfactory_dna.generation,
            'fitness_score': olfactory_dna.fitness_score
        }

    def _evolve_existing_dna(
        self,
        dna_id: str,
        feedback_brief: Any,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """기존 DNA 진화"""
        # DNA 라이브러리 준비
        dna_library = self.memory_dna_library

        # DNA가 메모리에 없으면 DB에서 로드 시도
        if dna_id not in dna_library and self.db:
            try:
                dna_model = self.db.query(OlfactoryDNAModel).filter_by(dna_id=dna_id).first()
                if dna_model:
                    # DB에서 로드한 DNA를 메모리에 추가 (간단한 구조로)
                    from fragrance_ai.models.living_scent.olfactory_recombinator import OlfactoryDNA
                    dna_library[dna_id] = OlfactoryDNA(
                        dna_id=dna_model.dna_id,
                        lineage=dna_model.lineage,
                        genotype={},  # 실제로는 변환 필요
                        phenotype_potential=dna_model.phenotype_potential,
                        story=dna_model.story,
                        creation_timestamp=str(dna_model.created_at),
                        generation=dna_model.generation,
                        fitness_score=dna_model.fitness_score
                    )
            except Exception as e:
                logger.error(f"Failed to load DNA from database: {e}")

        # 표현형 생성
        phenotype = self.epigenetic_variation.evolve(
            dna_id=dna_id,
            feedback_brief=feedback_brief,
            dna_library=dna_library
        )

        # 데이터베이스 저장
        self._save_phenotype_to_db(phenotype, user_id)

        return {
            'type': 'evolved_phenotype',
            'phenotype_id': phenotype.phenotype_id,
            'based_on_dna': phenotype.based_on_dna,
            'recipe': phenotype.recipe,
            'description': phenotype.description,
            'epigenetic_trigger': phenotype.epigenetic_trigger,
            'environmental_response': phenotype.environmental_response,
            'evolution_path': phenotype.evolution_path,
            'modifications': [
                {
                    'type': mod.marker_type.value,
                    'target': mod.target_gene,
                    'factor': mod.modification_factor
                }
                for mod in phenotype.modifications
            ]
        }

    def get_dna_info(self, dna_id: str) -> Optional[Dict[str, Any]]:
        """DNA 정보 조회"""
        # 메모리에서 먼저 확인
        if dna_id in self.memory_dna_library:
            dna = self.memory_dna_library[dna_id]
            return self.olfactory_recombinator.to_json(dna)

        # DB에서 조회
        if self.db:
            try:
                dna_model = self.db.query(OlfactoryDNAModel).filter_by(dna_id=dna_id).first()
                if dna_model:
                    return {
                        'dna_id': dna_model.dna_id,
                        'lineage': dna_model.lineage,
                        'story': dna_model.story,
                        'generation': dna_model.generation,
                        'fitness_score': dna_model.fitness_score,
                        'total_phenotypes': dna_model.total_phenotypes,
                        'average_rating': dna_model.average_rating
                    }
            except Exception as e:
                logger.error(f"Failed to get DNA info: {e}")

        return None

    def get_phenotype_info(self, phenotype_id: str) -> Optional[Dict[str, Any]]:
        """표현형 정보 조회"""
        # 메모리에서 먼저 확인
        if phenotype_id in self.memory_phenotype_library:
            phenotype = self.memory_phenotype_library[phenotype_id]
            return self.epigenetic_variation.to_json(phenotype)

        # DB에서 조회
        if self.db:
            try:
                pheno_model = self.db.query(ScentPhenotypeModel).filter_by(
                    phenotype_id=phenotype_id
                ).first()
                if pheno_model:
                    return {
                        'phenotype_id': pheno_model.phenotype_id,
                        'based_on_dna': pheno_model.based_on_dna,
                        'recipe': pheno_model.recipe,
                        'description': pheno_model.description,
                        'environmental_response': pheno_model.environmental_response,
                        'evolution_path': pheno_model.evolution_path
                    }
            except Exception as e:
                logger.error(f"Failed to get phenotype info: {e}")

        return None

    def get_evolution_tree(self, root_dna_id: str) -> Dict[str, Any]:
        """DNA의 전체 진화 트리 조회"""
        tree = {
            'root_dna': root_dna_id,
            'nodes': [],
            'total_generations': 0
        }

        # 메모리에서 관련 표현형 찾기
        for pheno_id, phenotype in self.memory_phenotype_library.items():
            if phenotype.based_on_dna == root_dna_id:
                tree['nodes'].append({
                    'id': pheno_id,
                    'type': 'phenotype',
                    'parent': phenotype.parent_phenotype,
                    'generation': len(phenotype.evolution_path)
                })

        tree['total_generations'] = max(
            [node['generation'] for node in tree['nodes']] + [0]
        )

        return tree


# 싱글톤 인스턴스
_orchestrator_instance = None

def get_living_scent_orchestrator(db_session: Optional[Session] = None) -> LivingScentOrchestrator:
    """싱글톤 LivingScentOrchestrator 인스턴스 반환"""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = LivingScentOrchestrator(db_session)
    return _orchestrator_instance