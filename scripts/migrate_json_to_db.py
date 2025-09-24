#!/usr/bin/env python3
"""
JSON 파일을 데이터베이스로 마이그레이션하는 스크립트

기존 분산된 JSON 파일들을 관계형 데이터베이스로 통합하여
데이터 일관성과 성능을 개선합니다.

사용법:
    python scripts/migrate_json_to_db.py --data-dir ./data --dry-run
    python scripts/migrate_json_to_db.py --data-dir ./data --execute
"""

import os
import sys
import json
import argparse
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent.parent))

from fragrance_ai.database.connection import initialize_database, get_db_session
from fragrance_ai.database.models import (
    FragranceNote, Recipe, RecipeIngredient, Brand, TrainingDataset,
    ModelCheckpoint, SearchLog, GenerationLog, RecipeEvaluation
)
from fragrance_ai.core.config import settings
from fragrance_ai.core.production_logging import get_logger

logger = get_logger(__name__)


@dataclass
class MigrationStats:
    """마이그레이션 통계"""
    fragrance_notes_created: int = 0
    recipes_created: int = 0
    brands_created: int = 0
    training_datasets_created: int = 0
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

    def add_error(self, error: str):
        """에러 추가"""
        self.errors.append(error)
        logger.error(f"Migration error: {error}")

    def get_summary(self) -> str:
        """마이그레이션 요약 반환"""
        return f"""
Migration Summary:
==================
Fragrance Notes: {self.fragrance_notes_created} created
Recipes: {self.recipes_created} created
Brands: {self.brands_created} created
Training Datasets: {self.training_datasets_created} created
Errors: {len(self.errors)}

Errors:
{chr(10).join(f"  - {error}" for error in self.errors) if self.errors else "  None"}
"""


class JSONDataMigrator:
    """JSON 데이터 마이그레이션 클래스"""

    def __init__(self, data_dir: Path, dry_run: bool = True):
        self.data_dir = data_dir
        self.dry_run = dry_run
        self.stats = MigrationStats()

        # 파일 매핑 정의
        self.file_mappings = {
            'fragrance_notes': [
                'fragrance_notes_database.json',
                'comprehensive_fragrance_notes_database.json',
                'initial/comprehensive_fragrance_ingredients.json',
                'initial/exotic_fragrance_ingredients.json'
            ],
            'recipes': [
                'fragrance_recipes_database.json',
                'artistic_fragrance_collection_templates.json'
            ],
            'brands': [
                'fragrance_brands_products.json'
            ],
            'training_datasets': [
                'advanced_training_dataset.json'
            ]
        }

    async def migrate_all(self) -> MigrationStats:
        """전체 마이그레이션 실행"""
        logger.info(f"Starting JSON to DB migration (dry_run={self.dry_run})")

        try:
            # 데이터베이스 초기화
            if not self.dry_run:
                initialize_database()

            # 순서대로 마이그레이션 실행
            await self._migrate_fragrance_notes()
            await self._migrate_brands()
            await self._migrate_recipes()
            await self._migrate_training_datasets()

            logger.info("Migration completed successfully")
            print(self.stats.get_summary())

        except Exception as e:
            error_msg = f"Migration failed: {str(e)}"
            self.stats.add_error(error_msg)
            logger.error(error_msg, exc_info=True)

        return self.stats

    def _load_json_files(self, file_patterns: List[str]) -> List[Dict[str, Any]]:
        """JSON 파일들을 로드"""
        all_data = []

        for pattern in file_patterns:
            file_path = self.data_dir / pattern
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # 데이터 구조 정규화
                    if isinstance(data, dict):
                        if 'fragrance_notes' in data:
                            # 중첩된 노트 구조 처리
                            notes_dict = data['fragrance_notes']
                            for category, category_data in notes_dict.items():
                                if 'notes' in category_data:
                                    for note_name, note_data in category_data['notes'].items():
                                        note_data['name'] = note_name
                                        note_data['fragrance_family'] = category
                                        all_data.append(note_data)
                        elif 'fragrance_recipes' in data:
                            all_data.extend(data['fragrance_recipes'])
                        elif 'notes' in data:
                            all_data.extend(data['notes'])
                        elif 'recipes' in data:
                            all_data.extend(data['recipes'])
                        elif 'brands' in data:
                            all_data.extend(data['brands'])
                        elif 'datasets' in data:
                            all_data.extend(data['datasets'])
                        else:
                            all_data.append(data)
                    elif isinstance(data, list):
                        all_data.extend(data)

                    logger.info(f"Loaded {len(data)} items from {file_path}")

                except Exception as e:
                    error_msg = f"Failed to load {file_path}: {str(e)}"
                    self.stats.add_error(error_msg)
            else:
                logger.warning(f"File not found: {file_path}")

        return all_data

    async def _migrate_fragrance_notes(self):
        """향료 노트 마이그레이션"""
        logger.info("Migrating fragrance notes...")

        raw_data = self._load_json_files(self.file_mappings['fragrance_notes'])

        if self.dry_run:
            logger.info(f"[DRY RUN] Would migrate {len(raw_data)} fragrance notes")
            self.stats.fragrance_notes_created = len(raw_data)
            return

        created_count = 0

        with get_db_session() as session:
            for item in raw_data:
                try:
                    # 중복 확인
                    existing = session.query(FragranceNote).filter_by(
                        name=item.get('name', ''),
                        name_english=item.get('name_english', item.get('name', ''))
                    ).first()

                    if existing:
                        logger.debug(f"Skipping duplicate note: {item.get('name')}")
                        continue

                    # FragranceNote 객체 생성
                    note = FragranceNote(
                        name=item.get('name', ''),
                        name_korean=item.get('korean_name', item.get('name_korean')),
                        name_english=item.get('english_name', item.get('name_english', item.get('name'))),
                        note_type=self._normalize_note_type(item.get('note_type', item.get('type', 'middle'))),
                        fragrance_family=item.get('fragrance_family', item.get('family')),
                        intensity=float(item.get('strength', item.get('intensity', 5.0))),
                        longevity=float(item.get('longevity', item.get('lasting_power', 5.0))),
                        sillage=float(item.get('sillage', item.get('projection', 5.0))),
                        description=item.get('description'),
                        description_korean=item.get('description_korean'),
                        origin=item.get('origin', item.get('source')),
                        extraction_method=item.get('extraction_method', item.get('method')),
                        mood_tags=item.get('mood', item.get('mood_tags', [])),
                        season_tags=item.get('season', item.get('season_tags', [])),
                        gender_tags=item.get('gender', item.get('gender_tags', [])),
                        price_per_ml=item.get('price_per_ml'),
                        supplier=item.get('supplier'),
                        grade=item.get('grade', 'standard'),
                        search_keywords=self._build_search_keywords(item)
                    )

                    session.add(note)
                    created_count += 1

                except Exception as e:
                    error_msg = f"Failed to create fragrance note {item.get('name', 'unknown')}: {str(e)}"
                    self.stats.add_error(error_msg)

            session.commit()
            logger.info(f"Created {created_count} fragrance notes")
            self.stats.fragrance_notes_created = created_count

    async def _migrate_brands(self):
        """브랜드 마이그레이션"""
        logger.info("Migrating brands...")

        raw_data = self._load_json_files(self.file_mappings['brands'])

        if self.dry_run:
            logger.info(f"[DRY RUN] Would migrate {len(raw_data)} brands")
            self.stats.brands_created = len(raw_data)
            return

        created_count = 0

        with get_db_session() as session:
            for item in raw_data:
                try:
                    # 중복 확인
                    existing = session.query(Brand).filter_by(
                        name=item.get('name', item.get('brand_name', ''))
                    ).first()

                    if existing:
                        logger.debug(f"Skipping duplicate brand: {item.get('name')}")
                        continue

                    # Brand 객체 생성
                    brand = Brand(
                        name=item.get('name', item.get('brand_name', '')),
                        name_korean=item.get('name_korean', item.get('korean_name')),
                        country=item.get('country', item.get('origin_country')),
                        founded_year=item.get('founded_year', item.get('established')),
                        brand_type=item.get('brand_type', item.get('type', 'commercial')),
                        description=item.get('description'),
                        description_korean=item.get('description_korean'),
                        heritage_story=item.get('heritage_story', item.get('story')),
                        signature_style=item.get('signature_style', item.get('style')),
                        price_range=item.get('price_range', 'mid-range'),
                        target_market=item.get('target_market', []),
                        website=item.get('website'),
                        logo_url=item.get('logo_url')
                    )

                    session.add(brand)
                    created_count += 1

                except Exception as e:
                    error_msg = f"Failed to create brand {item.get('name', 'unknown')}: {str(e)}"
                    self.stats.add_error(error_msg)

            session.commit()
            logger.info(f"Created {created_count} brands")
            self.stats.brands_created = created_count

    async def _migrate_recipes(self):
        """레시피 마이그레이션"""
        logger.info("Migrating recipes...")

        raw_data = self._load_json_files(self.file_mappings['recipes'])

        if self.dry_run:
            logger.info(f"[DRY RUN] Would migrate {len(raw_data)} recipes")
            self.stats.recipes_created = len(raw_data)
            return

        created_count = 0

        with get_db_session() as session:
            for item in raw_data:
                try:
                    # 중복 확인
                    existing = session.query(Recipe).filter_by(
                        name=item.get('name', '')
                    ).first()

                    if existing:
                        logger.debug(f"Skipping duplicate recipe: {item.get('name')}")
                        continue

                    # Recipe 객체 생성
                    recipe = Recipe(
                        name=item.get('name', ''),
                        name_korean=item.get('korean_name', item.get('name_korean')),
                        recipe_type=item.get('type', item.get('recipe_type', 'basic')),
                        fragrance_family=item.get('fragrance_family', item.get('family', 'floral')),
                        complexity=int(item.get('complexity', 5)),
                        estimated_cost=item.get('estimated_cost'),
                        batch_size_ml=int(item.get('batch_size_ml', item.get('batch_size', 100))),
                        description=item.get('description'),
                        description_korean=item.get('description_korean'),
                        concept=item.get('concept'),
                        target_audience=item.get('target_audience'),
                        generation_model=item.get('generation_model'),
                        generation_params=item.get('generation_params', {}),
                        quality_score=item.get('quality_score'),
                        sillage=float(item.get('sillage', 5.0)),
                        longevity=float(item.get('longevity', 5.0)),
                        complexity_rating=float(item.get('complexity_rating', 5.0)),
                        mood_tags=item.get('mood', item.get('mood_tags', [])),
                        season_tags=item.get('season', item.get('season_tags', [])),
                        gender_tags=[item.get('gender')] if isinstance(item.get('gender'), str) else item.get('gender', []),
                        status=item.get('status', 'draft'),
                        is_public=item.get('is_public', False),
                        production_notes=item.get('production_notes', {}),
                        maceration_time_days=item.get('maceration_time_days'),
                        aging_requirements=item.get('aging_requirements')
                    )

                    session.add(recipe)
                    session.flush()  # recipe.id를 얻기 위해

                    # 레시피 재료 처리
                    notes_data = item.get('notes', {})
                    ingredients = []

                    # 노트 구조에서 재료 추출
                    for position, notes_list in notes_data.items():
                        if isinstance(notes_list, list):
                            for note_item in notes_list:
                                ingredients.append({
                                    'note': note_item.get('note'),
                                    'percentage': note_item.get('percentage', 0),
                                    'note_position': position,
                                    'description': note_item.get('description')
                                })

                    # 기존 ingredients 필드도 확인
                    if not ingredients and item.get('ingredients'):
                        ingredients = item.get('ingredients')
                    elif not ingredients and item.get('composition'):
                        ingredients = item.get('composition')

                    await self._create_recipe_ingredients(session, recipe.id, ingredients)

                    created_count += 1

                except Exception as e:
                    error_msg = f"Failed to create recipe {item.get('name', 'unknown')}: {str(e)}"
                    self.stats.add_error(error_msg)

            session.commit()
            logger.info(f"Created {created_count} recipes")
            self.stats.recipes_created = created_count

    async def _create_recipe_ingredients(self, session, recipe_id: str, ingredients: List[Dict]):
        """레시피 재료 생성"""
        for ingredient_data in ingredients:
            try:
                # 노트 찾기 (이름으로)
                note_name = ingredient_data.get('note', ingredient_data.get('name', ''))
                note = session.query(FragranceNote).filter(
                    (FragranceNote.name == note_name) |
                    (FragranceNote.name_korean == note_name) |
                    (FragranceNote.name_english == note_name)
                ).first()

                if not note:
                    logger.warning(f"Note not found for ingredient: {note_name}")
                    continue

                ingredient = RecipeIngredient(
                    recipe_id=recipe_id,
                    note_id=note.id,
                    percentage=float(ingredient_data.get('percentage', ingredient_data.get('amount', 0))),
                    weight_grams=ingredient_data.get('weight_grams'),
                    role=ingredient_data.get('role', 'primary'),
                    note_position=ingredient_data.get('note_position', note.note_type),
                    notes=ingredient_data.get('notes'),
                    is_optional=ingredient_data.get('is_optional', False),
                    alternative_note_ids=ingredient_data.get('alternatives', [])
                )

                session.add(ingredient)

            except Exception as e:
                error_msg = f"Failed to create recipe ingredient {ingredient_data}: {str(e)}"
                self.stats.add_error(error_msg)

    async def _migrate_training_datasets(self):
        """훈련 데이터셋 마이그레이션"""
        logger.info("Migrating training datasets...")

        raw_data = self._load_json_files(self.file_mappings['training_datasets'])

        if self.dry_run:
            logger.info(f"[DRY RUN] Would migrate {len(raw_data)} training datasets")
            self.stats.training_datasets_created = len(raw_data)
            return

        created_count = 0

        with get_db_session() as session:
            for item in raw_data:
                try:
                    # 중복 확인
                    existing = session.query(TrainingDataset).filter_by(
                        name=item.get('name', '')
                    ).first()

                    if existing:
                        logger.debug(f"Skipping duplicate training dataset: {item.get('name')}")
                        continue

                    # TrainingDataset 객체 생성
                    dataset = TrainingDataset(
                        name=item.get('name', ''),
                        description=item.get('description'),
                        dataset_type=item.get('dataset_type', item.get('type', 'recipe_generation')),
                        version=item.get('version', '1.0'),
                        total_samples=int(item.get('total_samples', 0)),
                        training_samples=int(item.get('training_samples', 0)),
                        validation_samples=int(item.get('validation_samples', 0)),
                        test_samples=int(item.get('test_samples', 0)),
                        data_path=item.get('data_path'),
                        metadata_path=item.get('metadata_path'),
                        status=item.get('status', 'preparing'),
                        quality_score=item.get('quality_score'),
                        creation_method=item.get('creation_method', 'manual'),
                        source_info=item.get('source_info', {})
                    )

                    session.add(dataset)
                    created_count += 1

                except Exception as e:
                    error_msg = f"Failed to create training dataset {item.get('name', 'unknown')}: {str(e)}"
                    self.stats.add_error(error_msg)

            session.commit()
            logger.info(f"Created {created_count} training datasets")
            self.stats.training_datasets_created = created_count

    def _normalize_note_type(self, note_type: str) -> str:
        """노트 타입 정규화"""
        if not note_type:
            return 'middle'

        note_type = note_type.lower()
        if note_type in ['top', 'head', 'opening']:
            return 'top'
        elif note_type in ['middle', 'heart', 'body']:
            return 'middle'
        elif note_type in ['base', 'bottom', 'foundation']:
            return 'base'
        else:
            return 'middle'

    def _build_search_keywords(self, item: Dict[str, Any]) -> str:
        """검색 키워드 구축"""
        keywords = []

        # 이름들
        for key in ['name', 'name_korean', 'name_english', 'korean_name', 'english_name']:
            if item.get(key):
                keywords.append(str(item[key]))

        # 태그들
        for key in ['mood_tags', 'season_tags', 'gender_tags', 'moods', 'seasons', 'gender']:
            if item.get(key) and isinstance(item[key], list):
                keywords.extend(item[key])

        # 기타
        for key in ['fragrance_family', 'family', 'origin', 'source']:
            if item.get(key):
                keywords.append(str(item[key]))

        return ' '.join(set(keywords))


async def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='Migrate JSON data to database')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Directory containing JSON files')
    parser.add_argument('--dry-run', action='store_true',
                       help='Run in dry-run mode (no actual changes)')
    parser.add_argument('--execute', action='store_true',
                       help='Execute the migration (actual changes)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    # 로깅 설정
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 실행 모드 결정
    if args.execute and args.dry_run:
        print("ERROR: Cannot use both --execute and --dry-run")
        sys.exit(1)

    dry_run = not args.execute  # 기본값은 dry-run

    if dry_run:
        print("Running in DRY-RUN mode - no changes will be made")
    else:
        print("Running in EXECUTE mode - database will be modified")
        response = input("Are you sure? (yes/no): ")
        if response.lower() != 'yes':
            print("Migration cancelled")
            sys.exit(0)

    # 데이터 디렉토리 확인
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)

    # 마이그레이션 실행
    migrator = JSONDataMigrator(data_dir, dry_run=dry_run)
    stats = await migrator.migrate_all()

    # 결과 출력
    if stats.errors:
        print(f"\nMigration completed with {len(stats.errors)} errors")
        sys.exit(1)
    else:
        print(f"\nMigration completed successfully")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())