"""
레시피 저장소

향수 레시피 데이터에 특화된 쿼리와 비즈니스 로직을 제공합니다.
"""

from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy import func, and_, or_, desc, asc
from sqlalchemy.orm import Session, joinedload

from .base import BaseRepository
from ..database.models import Recipe, RecipeIngredient, FragranceNote, RecipeEvaluation
from ..core.production_logging import get_logger

logger = get_logger(__name__)


class RecipeRepository(BaseRepository[Recipe]):
    """레시피 저장소"""

    def __init__(self, session: Session):
        super().__init__(Recipe, session)

    # ==========================================
    # 레시피 특화 쿼리
    # ==========================================

    def get_recipe_with_ingredients(self, recipe_id: str) -> Optional[Recipe]:
        """재료와 함께 레시피 조회"""
        try:
            recipe = self.session.query(Recipe).options(
                joinedload(Recipe.ingredients).joinedload(RecipeIngredient.note)
            ).filter(Recipe.id == recipe_id).first()

            if recipe:
                logger.debug(f"Found recipe {recipe_id} with {len(recipe.ingredients)} ingredients")

            return recipe

        except Exception as e:
            logger.error(f"Failed to get recipe with ingredients {recipe_id}: {str(e)}")
            raise

    def get_recipes_by_fragrance_family(self, family: str, limit: Optional[int] = None) -> List[Recipe]:
        """향족별 레시피 조회"""
        try:
            query = self.session.query(Recipe).filter(Recipe.fragrance_family == family)

            if limit:
                query = query.limit(limit)

            recipes = query.order_by(desc(Recipe.quality_score)).all()
            logger.debug(f"Found {len(recipes)} recipes for fragrance family: {family}")
            return recipes

        except Exception as e:
            logger.error(f"Failed to get recipes by fragrance family {family}: {str(e)}")
            raise

    def search_recipes_by_name(self, name: str, exact: bool = False) -> List[Recipe]:
        """이름으로 레시피 검색"""
        try:
            query = self.session.query(Recipe)

            if exact:
                query = query.filter(
                    or_(
                        Recipe.name == name,
                        Recipe.name_korean == name
                    )
                )
            else:
                search_pattern = f"%{name}%"
                query = query.filter(
                    or_(
                        Recipe.name.ilike(search_pattern),
                        Recipe.name_korean.ilike(search_pattern),
                        Recipe.description.ilike(search_pattern),
                        Recipe.description_korean.ilike(search_pattern)
                    )
                )

            recipes = query.order_by(Recipe.name).all()
            logger.debug(f"Found {len(recipes)} recipes for name search: {name}")
            return recipes

        except Exception as e:
            logger.error(f"Failed to search recipes by name {name}: {str(e)}")
            raise

    def find_recipes_by_tags(self,
                           mood_tags: Optional[List[str]] = None,
                           season_tags: Optional[List[str]] = None,
                           gender_tags: Optional[List[str]] = None,
                           match_all: bool = False) -> List[Recipe]:
        """태그로 레시피 검색"""
        try:
            query = self.session.query(Recipe)
            conditions = []

            # 무드 태그 필터
            if mood_tags:
                if match_all:
                    for tag in mood_tags:
                        conditions.append(
                            Recipe.mood_tags.op('JSON_CONTAINS')(f'"{tag}"')
                        )
                else:
                    mood_conditions = [
                        Recipe.mood_tags.op('JSON_CONTAINS')(f'"{tag}"')
                        for tag in mood_tags
                    ]
                    conditions.append(or_(*mood_conditions))

            # 시즌 태그 필터
            if season_tags:
                if match_all:
                    for tag in season_tags:
                        conditions.append(
                            Recipe.season_tags.op('JSON_CONTAINS')(f'"{tag}"')
                        )
                else:
                    season_conditions = [
                        Recipe.season_tags.op('JSON_CONTAINS')(f'"{tag}"')
                        for tag in season_tags
                    ]
                    conditions.append(or_(*season_conditions))

            # 성별 태그 필터
            if gender_tags:
                if match_all:
                    for tag in gender_tags:
                        conditions.append(
                            Recipe.gender_tags.op('JSON_CONTAINS')(f'"{tag}"')
                        )
                else:
                    gender_conditions = [
                        Recipe.gender_tags.op('JSON_CONTAINS')(f'"{tag}"')
                        for tag in gender_tags
                    ]
                    conditions.append(or_(*gender_conditions))

            # 조건 적용
            if conditions:
                if match_all:
                    query = query.filter(and_(*conditions))
                else:
                    query = query.filter(or_(*conditions))

            recipes = query.order_by(desc(Recipe.quality_score)).all()
            logger.debug(f"Found {len(recipes)} recipes by tags")
            return recipes

        except Exception as e:
            logger.error(f"Failed to search recipes by tags: {str(e)}")
            raise

    def find_recipes_by_complexity(self,
                                  min_complexity: Optional[int] = None,
                                  max_complexity: Optional[int] = None) -> List[Recipe]:
        """복잡도로 레시피 검색"""
        try:
            query = self.session.query(Recipe)

            if min_complexity is not None:
                query = query.filter(Recipe.complexity >= min_complexity)
            if max_complexity is not None:
                query = query.filter(Recipe.complexity <= max_complexity)

            recipes = query.order_by(Recipe.complexity.asc()).all()
            logger.debug(f"Found {len(recipes)} recipes by complexity")
            return recipes

        except Exception as e:
            logger.error(f"Failed to search recipes by complexity: {str(e)}")
            raise

    def find_recipes_by_characteristics(self,
                                      sillage_min: Optional[float] = None,
                                      sillage_max: Optional[float] = None,
                                      longevity_min: Optional[float] = None,
                                      longevity_max: Optional[float] = None,
                                      complexity_rating_min: Optional[float] = None,
                                      complexity_rating_max: Optional[float] = None) -> List[Recipe]:
        """특성으로 레시피 검색"""
        try:
            query = self.session.query(Recipe)

            # 확산성 필터
            if sillage_min is not None:
                query = query.filter(Recipe.sillage >= sillage_min)
            if sillage_max is not None:
                query = query.filter(Recipe.sillage <= sillage_max)

            # 지속성 필터
            if longevity_min is not None:
                query = query.filter(Recipe.longevity >= longevity_min)
            if longevity_max is not None:
                query = query.filter(Recipe.longevity <= longevity_max)

            # 복잡도 평가 필터
            if complexity_rating_min is not None:
                query = query.filter(Recipe.complexity_rating >= complexity_rating_min)
            if complexity_rating_max is not None:
                query = query.filter(Recipe.complexity_rating <= complexity_rating_max)

            recipes = query.order_by(desc(Recipe.quality_score)).all()
            logger.debug(f"Found {len(recipes)} recipes by characteristics")
            return recipes

        except Exception as e:
            logger.error(f"Failed to search recipes by characteristics: {str(e)}")
            raise

    def get_recipes_by_status(self, status: str) -> List[Recipe]:
        """상태별 레시피 조회"""
        return self.find_by(status=status)

    def get_public_recipes(self, limit: Optional[int] = None) -> List[Recipe]:
        """공개 레시피 조회"""
        try:
            query = self.session.query(Recipe).filter(Recipe.is_public == True)

            if limit:
                query = query.limit(limit)

            recipes = query.order_by(desc(Recipe.quality_score)).all()
            logger.debug(f"Found {len(recipes)} public recipes")
            return recipes

        except Exception as e:
            logger.error(f"Failed to get public recipes: {str(e)}")
            raise

    def get_top_rated_recipes(self, limit: int = 20) -> List[Recipe]:
        """최고 평점 레시피 조회"""
        try:
            recipes = self.session.query(Recipe).filter(
                Recipe.quality_score.is_not(None)
            ).order_by(
                desc(Recipe.quality_score)
            ).limit(limit).all()

            logger.debug(f"Found {len(recipes)} top rated recipes")
            return recipes

        except Exception as e:
            logger.error(f"Failed to get top rated recipes: {str(e)}")
            raise

    def find_recipes_containing_note(self, note_id: str) -> List[Recipe]:
        """특정 노트를 포함하는 레시피 조회"""
        try:
            recipes = self.session.query(Recipe).join(
                RecipeIngredient, Recipe.id == RecipeIngredient.recipe_id
            ).filter(
                RecipeIngredient.note_id == note_id
            ).order_by(desc(Recipe.quality_score)).all()

            logger.debug(f"Found {len(recipes)} recipes containing note {note_id}")
            return recipes

        except Exception as e:
            logger.error(f"Failed to find recipes containing note {note_id}: {str(e)}")
            raise

    def find_similar_recipes(self, recipe_id: str, limit: int = 10) -> List[Recipe]:
        """유사한 레시피 찾기"""
        try:
            # 기준 레시피 조회
            base_recipe = self.get_recipe_with_ingredients(recipe_id)
            if not base_recipe:
                return []

            # 같은 향족의 레시피들 중에서 유사한 특성을 가진 것들 찾기
            similar_recipes = self.session.query(Recipe).filter(
                and_(
                    Recipe.id != recipe_id,
                    Recipe.fragrance_family == base_recipe.fragrance_family,
                    func.abs(Recipe.sillage - base_recipe.sillage) <= 1.0,
                    func.abs(Recipe.longevity - base_recipe.longevity) <= 1.0,
                    func.abs(Recipe.complexity_rating - base_recipe.complexity_rating) <= 1.0
                )
            ).order_by(
                func.abs(Recipe.sillage - base_recipe.sillage) +
                func.abs(Recipe.longevity - base_recipe.longevity) +
                func.abs(Recipe.complexity_rating - base_recipe.complexity_rating)
            ).limit(limit).all()

            logger.debug(f"Found {len(similar_recipes)} similar recipes for {recipe_id}")
            return similar_recipes

        except Exception as e:
            logger.error(f"Failed to find similar recipes for {recipe_id}: {str(e)}")
            raise

    # ==========================================
    # 레시피 재료 관리
    # ==========================================

    def add_ingredient_to_recipe(self,
                                recipe_id: str,
                                note_id: str,
                                percentage: float,
                                role: str = 'primary',
                                note_position: Optional[str] = None,
                                **kwargs) -> RecipeIngredient:
        """레시피에 재료 추가"""
        try:
            # 기존 재료 확인
            existing = self.session.query(RecipeIngredient).filter(
                and_(
                    RecipeIngredient.recipe_id == recipe_id,
                    RecipeIngredient.note_id == note_id
                )
            ).first()

            if existing:
                raise ValueError(f"Ingredient {note_id} already exists in recipe {recipe_id}")

            # 노트 정보 확인
            note = self.session.query(FragranceNote).filter(
                FragranceNote.id == note_id
            ).first()

            if not note:
                raise ValueError(f"Note {note_id} not found")

            # note_position 기본값 설정
            if not note_position:
                note_position = note.note_type

            ingredient = RecipeIngredient(
                recipe_id=recipe_id,
                note_id=note_id,
                percentage=percentage,
                role=role,
                note_position=note_position,
                **kwargs
            )

            self.session.add(ingredient)
            self.session.flush()

            logger.debug(f"Added ingredient {note_id} to recipe {recipe_id}")
            return ingredient

        except Exception as e:
            logger.error(f"Failed to add ingredient to recipe: {str(e)}")
            self.session.rollback()
            raise

    def remove_ingredient_from_recipe(self, recipe_id: str, note_id: str) -> bool:
        """레시피에서 재료 제거"""
        try:
            ingredient = self.session.query(RecipeIngredient).filter(
                and_(
                    RecipeIngredient.recipe_id == recipe_id,
                    RecipeIngredient.note_id == note_id
                )
            ).first()

            if not ingredient:
                return False

            self.session.delete(ingredient)
            self.session.flush()

            logger.debug(f"Removed ingredient {note_id} from recipe {recipe_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to remove ingredient from recipe: {str(e)}")
            self.session.rollback()
            raise

    def update_ingredient_percentage(self,
                                   recipe_id: str,
                                   note_id: str,
                                   new_percentage: float) -> bool:
        """재료 농도 업데이트"""
        try:
            ingredient = self.session.query(RecipeIngredient).filter(
                and_(
                    RecipeIngredient.recipe_id == recipe_id,
                    RecipeIngredient.note_id == note_id
                )
            ).first()

            if not ingredient:
                return False

            ingredient.percentage = new_percentage
            self.session.flush()

            logger.debug(f"Updated ingredient {note_id} percentage to {new_percentage}% in recipe {recipe_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update ingredient percentage: {str(e)}")
            self.session.rollback()
            raise

    def get_recipe_ingredients_by_position(self, recipe_id: str, note_position: str) -> List[RecipeIngredient]:
        """포지션별 레시피 재료 조회"""
        try:
            ingredients = self.session.query(RecipeIngredient).options(
                joinedload(RecipeIngredient.note)
            ).filter(
                and_(
                    RecipeIngredient.recipe_id == recipe_id,
                    RecipeIngredient.note_position == note_position
                )
            ).order_by(desc(RecipeIngredient.percentage)).all()

            logger.debug(f"Found {len(ingredients)} {note_position} ingredients for recipe {recipe_id}")
            return ingredients

        except Exception as e:
            logger.error(f"Failed to get recipe ingredients by position: {str(e)}")
            raise

    def validate_recipe_composition(self, recipe_id: str) -> Dict[str, Any]:
        """레시피 구성 검증"""
        try:
            recipe = self.get_recipe_with_ingredients(recipe_id)
            if not recipe:
                return {"valid": False, "error": "Recipe not found"}

            # 총 농도 계산
            total_percentage = sum(ingredient.percentage for ingredient in recipe.ingredients)

            # 포지션별 분포
            position_distribution = {}
            for ingredient in recipe.ingredients:
                position = ingredient.note_position
                if position not in position_distribution:
                    position_distribution[position] = 0
                position_distribution[position] += ingredient.percentage

            # 검증 결과
            validation = {
                "valid": True,
                "total_percentage": total_percentage,
                "position_distribution": position_distribution,
                "warnings": [],
                "errors": []
            }

            # 검증 규칙
            if total_percentage < 99 or total_percentage > 101:
                validation["errors"].append(f"Total percentage should be 100%, got {total_percentage:.2f}%")
                validation["valid"] = False

            if position_distribution.get("top", 0) < 10:
                validation["warnings"].append("Top notes percentage is very low (< 10%)")

            if position_distribution.get("base", 0) < 15:
                validation["warnings"].append("Base notes percentage is low (< 15%)")

            if len(recipe.ingredients) < 3:
                validation["warnings"].append("Recipe has very few ingredients (< 3)")

            logger.debug(f"Recipe {recipe_id} validation completed")
            return validation

        except Exception as e:
            logger.error(f"Failed to validate recipe composition: {str(e)}")
            raise

    # ==========================================
    # 통계 및 분석
    # ==========================================

    def get_fragrance_family_distribution(self) -> Dict[str, int]:
        """향족별 레시피 분포"""
        try:
            results = self.session.query(
                Recipe.fragrance_family,
                func.count(Recipe.id)
            ).group_by(
                Recipe.fragrance_family
            ).all()

            distribution = {family: count for family, count in results if family}
            logger.debug(f"Recipe fragrance family distribution: {distribution}")
            return distribution

        except Exception as e:
            logger.error(f"Failed to get recipe fragrance family distribution: {str(e)}")
            raise

    def get_complexity_distribution(self) -> Dict[str, int]:
        """복잡도별 분포 통계"""
        try:
            results = self.session.query(
                Recipe.complexity,
                func.count(Recipe.id)
            ).group_by(
                Recipe.complexity
            ).order_by(Recipe.complexity).all()

            distribution = {f"complexity_{complexity}": count for complexity, count in results}
            logger.debug(f"Recipe complexity distribution: {distribution}")
            return distribution

        except Exception as e:
            logger.error(f"Failed to get recipe complexity distribution: {str(e)}")
            raise

    def get_most_used_notes(self, limit: int = 20) -> List[Tuple[FragranceNote, int]]:
        """가장 많이 사용되는 노트 조회"""
        try:
            results = self.session.query(
                FragranceNote,
                func.count(RecipeIngredient.id).label('usage_count')
            ).join(
                RecipeIngredient, FragranceNote.id == RecipeIngredient.note_id
            ).group_by(
                FragranceNote.id
            ).order_by(
                desc(func.count(RecipeIngredient.id))
            ).limit(limit).all()

            logger.debug(f"Found {len(results)} most used notes")
            return [(note, count) for note, count in results]

        except Exception as e:
            logger.error(f"Failed to get most used notes: {str(e)}")
            raise

    def get_recipe_statistics(self) -> Dict[str, Any]:
        """레시피 통계 요약"""
        try:
            total_recipes = self.count()
            public_recipes = self.count({"is_public": True})
            approved_recipes = self.count({"status": "approved"})

            # 평균 품질 점수
            avg_quality = self.session.query(
                func.avg(Recipe.quality_score)
            ).filter(Recipe.quality_score.is_not(None)).scalar()

            # 평균 복잡도
            avg_complexity = self.session.query(
                func.avg(Recipe.complexity)
            ).scalar()

            stats = {
                "total_recipes": total_recipes,
                "public_recipes": public_recipes,
                "approved_recipes": approved_recipes,
                "average_quality_score": float(avg_quality or 0),
                "average_complexity": float(avg_complexity or 0),
                "fragrance_family_distribution": self.get_fragrance_family_distribution(),
                "complexity_distribution": self.get_complexity_distribution()
            }

            logger.debug("Generated recipe statistics")
            return stats

        except Exception as e:
            logger.error(f"Failed to get recipe statistics: {str(e)}")
            raise

    # ==========================================
    # 고급 검색
    # ==========================================

    def advanced_search(self,
                       query_text: Optional[str] = None,
                       families: Optional[List[str]] = None,
                       complexity_range: Optional[Tuple[int, int]] = None,
                       quality_min: Optional[float] = None,
                       tags: Optional[List[str]] = None,
                       status: Optional[str] = None,
                       is_public: Optional[bool] = None,
                       contains_notes: Optional[List[str]] = None,
                       excludes_notes: Optional[List[str]] = None,
                       sort_by: str = "quality_score",
                       sort_desc: bool = True,
                       limit: int = 50,
                       offset: int = 0) -> List[Recipe]:
        """고급 통합 검색"""
        try:
            query = self.session.query(Recipe)

            # 텍스트 검색
            if query_text:
                search_pattern = f"%{query_text}%"
                query = query.filter(
                    or_(
                        Recipe.name.ilike(search_pattern),
                        Recipe.name_korean.ilike(search_pattern),
                        Recipe.description.ilike(search_pattern),
                        Recipe.description_korean.ilike(search_pattern),
                        Recipe.concept.ilike(search_pattern)
                    )
                )

            # 향족 필터
            if families:
                query = query.filter(Recipe.fragrance_family.in_(families))

            # 복잡도 범위
            if complexity_range:
                query = query.filter(
                    Recipe.complexity.between(complexity_range[0], complexity_range[1])
                )

            # 품질 점수 최소값
            if quality_min is not None:
                query = query.filter(Recipe.quality_score >= quality_min)

            # 상태 필터
            if status:
                query = query.filter(Recipe.status == status)

            # 공개 여부
            if is_public is not None:
                query = query.filter(Recipe.is_public == is_public)

            # 특정 노트 포함 필터
            if contains_notes:
                for note_id in contains_notes:
                    query = query.filter(
                        Recipe.ingredients.any(RecipeIngredient.note_id == note_id)
                    )

            # 특정 노트 제외 필터
            if excludes_notes:
                for note_id in excludes_notes:
                    query = query.filter(
                        ~Recipe.ingredients.any(RecipeIngredient.note_id == note_id)
                    )

            # 태그 필터
            if tags:
                tag_conditions = []
                for tag in tags:
                    tag_conditions.extend([
                        Recipe.mood_tags.op('JSON_CONTAINS')(f'"{tag}"'),
                        Recipe.season_tags.op('JSON_CONTAINS')(f'"{tag}"'),
                        Recipe.gender_tags.op('JSON_CONTAINS')(f'"{tag}"')
                    ])
                query = query.filter(or_(*tag_conditions))

            # 정렬
            if hasattr(Recipe, sort_by):
                sort_column = getattr(Recipe, sort_by)
                query = query.order_by(desc(sort_column) if sort_desc else asc(sort_column))

            # 페이징
            query = query.offset(offset).limit(limit)

            recipes = query.all()
            logger.debug(f"Advanced search returned {len(recipes)} recipes")
            return recipes

        except Exception as e:
            logger.error(f"Failed to perform advanced recipe search: {str(e)}")
            raise