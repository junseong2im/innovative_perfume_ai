#!/usr/bin/env python3
"""
향료 데이터베이스 초기화 스크립트
전 세계 향료 원료 데이터를 데이터베이스에 삽입
"""
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from fragrance_ai.core.config import settings
from fragrance_ai.database.models import FragranceIngredient
from fragrance_ai.database.base import Base


def load_fragrance_data() -> List[Dict[str, Any]]:
    """모든 향료 데이터 파일을 로드하여 통합"""
    data_dir = Path(__file__).parent.parent / "data" / "initial"
    all_ingredients = []

    # 모든 향료 데이터 파일들
    data_files = [
        "sample_fragrances.json",
        "comprehensive_fragrance_ingredients.json",
        "extended_fragrance_ingredients.json",
        "specialty_fragrance_ingredients.json",
        "exotic_fragrance_ingredients.json"
    ]

    for file_name in data_files:
        file_path = data_dir / file_name
        if file_path.exists():
            print(f"Loading {file_name}...")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_ingredients.extend(data)
                else:
                    print(f"Warning: {file_name} does not contain a list")
        else:
            print(f"Warning: {file_name} not found")

    print(f"Loaded total {len(all_ingredients)} ingredients")
    return all_ingredients


def clean_and_validate_data(ingredients: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """데이터 정리 및 유효성 검증"""
    cleaned_ingredients = []
    seen_names = set()

    for ingredient in ingredients:
        # 필수 필드 확인
        if not ingredient.get('name') or not ingredient.get('english_name'):
            print(f"Skipping ingredient with missing name: {ingredient}")
            continue

        # 중복 제거
        name_key = ingredient['name'].lower()
        if name_key in seen_names:
            print(f"Skipping duplicate ingredient: {ingredient['name']}")
            continue
        seen_names.add(name_key)

        # 데이터 타입 변환 및 기본값 설정
        cleaned_ingredient = {
            'name': ingredient.get('name', '').strip(),
            'english_name': ingredient.get('english_name', '').strip(),
            'korean_name': ingredient.get('korean_name', '').strip(),
            'category': ingredient.get('category', 'Unknown'),
            'fragrance_family': ingredient.get('fragrance_family', 'other'),
            'note_type': ingredient.get('note_type', 'middle'),
            'description': ingredient.get('description', ''),
            'origin': ingredient.get('origin', ''),
            'cas_number': ingredient.get('cas_number'),
            'intensity': float(ingredient.get('intensity', 5.0)) if ingredient.get('intensity') else 5.0,
            'longevity': float(ingredient.get('longevity', 5.0)) if ingredient.get('longevity') else 5.0,
            'sillage': float(ingredient.get('sillage', 5.0)) if ingredient.get('sillage') else 5.0,
            'price_range': ingredient.get('price_range', 'moderate'),
            'safety_rating': ingredient.get('safety_rating', 'safe'),
            'allergen_info': ingredient.get('allergen_info', ''),
            'blending_guidelines': ingredient.get('blending_guidelines', ''),
            'supplier_info': ingredient.get('supplier_info', ''),
            'molecular_formula': ingredient.get('molecular_formula', ''),
        }

        cleaned_ingredients.append(cleaned_ingredient)

    print(f"Cleaned and validated {len(cleaned_ingredients)} ingredients")
    return cleaned_ingredients


def create_database_tables(engine):
    """데이터베이스 테이블 생성"""
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully")


def insert_ingredients(session, ingredients: List[Dict[str, Any]]):
    """향료 데이터를 데이터베이스에 삽입"""
    print(f"Inserting {len(ingredients)} ingredients into database...")

    # 기존 데이터 확인
    existing_count = session.query(FragranceIngredient).count()
    if existing_count > 0:
        print(f"Found {existing_count} existing ingredients in database")
        response = input("Do you want to clear existing data? (y/N): ")
        if response.lower() == 'y':
            session.query(FragranceIngredient).delete()
            session.commit()
            print("Existing data cleared")

    # 배치 삽입
    batch_size = 100
    inserted_count = 0

    for i in range(0, len(ingredients), batch_size):
        batch = ingredients[i:i + batch_size]

        try:
            # FragranceIngredient 객체 생성
            ingredient_objects = []
            for ingredient_data in batch:
                ingredient = FragranceIngredient(
                    name=ingredient_data['name'],
                    english_name=ingredient_data['english_name'],
                    korean_name=ingredient_data['korean_name'],
                    category=ingredient_data['category'],
                    fragrance_family=ingredient_data['fragrance_family'],
                    note_type=ingredient_data['note_type'],
                    description=ingredient_data['description'],
                    origin=ingredient_data['origin'],
                    cas_number=ingredient_data['cas_number'],
                    intensity=ingredient_data['intensity'],
                    longevity=ingredient_data['longevity'],
                    sillage=ingredient_data['sillage'],
                    price_range=ingredient_data['price_range'],
                    safety_rating=ingredient_data['safety_rating'],
                    allergen_info=ingredient_data['allergen_info'],
                    blending_guidelines=ingredient_data['blending_guidelines'],
                    supplier_info=ingredient_data['supplier_info'],
                    molecular_formula=ingredient_data['molecular_formula'],
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                ingredient_objects.append(ingredient)

            # 데이터베이스에 삽입
            session.add_all(ingredient_objects)
            session.commit()

            inserted_count += len(batch)
            print(f"Inserted batch {i//batch_size + 1}: {inserted_count}/{len(ingredients)} ingredients")

        except Exception as e:
            session.rollback()
            print(f"Error inserting batch {i//batch_size + 1}: {e}")
            continue

    print(f"Successfully inserted {inserted_count} ingredients")


def create_database_statistics(session):
    """데이터베이스 통계 생성"""
    print("\nGenerating database statistics...")

    # 총 향료 수
    total_count = session.query(FragranceIngredient).count()
    print(f"Total ingredients: {total_count}")

    # 카테고리별 통계
    category_stats = session.execute(text("""
        SELECT category, COUNT(*) as count
        FROM fragrance_ingredients
        GROUP BY category
        ORDER BY count DESC
    """)).fetchall()

    print("\nIngredients by category:")
    for category, count in category_stats:
        print(f"  {category}: {count}")

    # 향료 패밀리별 통계
    family_stats = session.execute(text("""
        SELECT fragrance_family, COUNT(*) as count
        FROM fragrance_ingredients
        GROUP BY fragrance_family
        ORDER BY count DESC
    """)).fetchall()

    print("\nIngredients by fragrance family:")
    for family, count in family_stats:
        print(f"  {family}: {count}")

    # 노트 타입별 통계
    note_stats = session.execute(text("""
        SELECT note_type, COUNT(*) as count
        FROM fragrance_ingredients
        GROUP BY note_type
        ORDER BY count DESC
    """)).fetchall()

    print("\nIngredients by note type:")
    for note_type, count in note_stats:
        print(f"  {note_type}: {count}")

    # 가격대별 통계
    price_stats = session.execute(text("""
        SELECT price_range, COUNT(*) as count
        FROM fragrance_ingredients
        GROUP BY price_range
        ORDER BY count DESC
    """)).fetchall()

    print("\nIngredients by price range:")
    for price_range, count in price_stats:
        print(f"  {price_range}: {count}")


def main():
    """메인 함수"""
    print("="*60)
    print("Fragrance Database Population Script")
    print("="*60)

    try:
        # 1. 데이터 로드
        ingredients_data = load_fragrance_data()
        if not ingredients_data:
            print("No ingredient data found!")
            return

        # 2. 데이터 정리 및 검증
        cleaned_data = clean_and_validate_data(ingredients_data)
        if not cleaned_data:
            print("No valid ingredient data after cleaning!")
            return

        # 3. 데이터베이스 연결
        print(f"Connecting to database: {settings.database_url}")
        engine = create_engine(settings.database_url)

        # 4. 테이블 생성
        create_database_tables(engine)

        # 5. 세션 생성
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        session = SessionLocal()

        try:
            # 6. 데이터 삽입
            insert_ingredients(session, cleaned_data)

            # 7. 통계 생성
            create_database_statistics(session)

        finally:
            session.close()

        print("\n" + "="*60)
        print("Database population completed successfully!")
        print("="*60)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()