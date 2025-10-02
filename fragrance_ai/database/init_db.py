"""
PostgreSQL 데이터베이스 초기화 및 기본 데이터 로드
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sentence_transformers import SentenceTransformer

from fragrance_ai.database.schema import (
    Base, Fragrance, Note, FragranceComposition,
    BlendingRule, AccordTemplate, KnowledgeBase
)


class DatabaseInitializer:
    """데이터베이스 초기화 클래스"""

    def __init__(self, database_url: str = None):
        if database_url is None:
            database_url = os.getenv(
                "DATABASE_URL",
                "postgresql://postgres:password@localhost:5432/fragrance_ai"
            )

        self.engine = create_engine(database_url)
        self.Session = sessionmaker(bind=self.engine)

        # 임베딩 모델 (한 번만 로드)
        print("임베딩 모델 로드 중...")
        self.encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def initialize(self):
        """전체 초기화 프로세스"""
        print("데이터베이스 초기화 시작...")

        # 1. pgvector 확장 설치
        self.setup_pgvector()

        # 2. 테이블 생성
        self.create_tables()

        # 3. 기본 데이터 로드
        self.load_initial_data()

        print("데이터베이스 초기화 완료!")

    def setup_pgvector(self):
        """pgvector 확장 설치"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.commit()
                print("pgvector 확장 설치 완료")
        except Exception as e:
            print(f"pgvector 설치 경고: {e}")

    def create_tables(self):
        """모든 테이블 생성"""
        Base.metadata.create_all(self.engine)
        print("테이블 생성 완료")

    def load_initial_data(self):
        """초기 데이터 로드"""
        session = self.Session()
        try:
            # 1. 기본 향료 노트 데이터
            self.load_notes(session)

            # 2. 블렌딩 규칙
            self.load_blending_rules(session)

            # 3. 어코드 템플릿
            self.load_accord_templates(session)

            # 4. 지식베이스
            self.load_knowledge_base(session)

            # 5. 샘플 향수 데이터
            self.load_sample_fragrances(session)

            session.commit()
            print("초기 데이터 로드 완료")

        except Exception as e:
            session.rollback()
            print(f"데이터 로드 실패: {e}")
            raise
        finally:
            session.close()

    def load_notes(self, session):
        """기본 향료 노트 데이터 로드"""
        notes_data = [
            # Top Notes (높은 휘발성)
            {"name": "Bergamot", "type": "citrus", "pyramid_level": "top",
             "volatility": 0.95, "strength": 0.7, "longevity": 0.2,
             "description": "Fresh, citrusy with slight floral undertone",
             "origin": "Italy", "is_natural": True},

            {"name": "Lemon", "type": "citrus", "pyramid_level": "top",
             "volatility": 0.98, "strength": 0.6, "longevity": 0.15,
             "description": "Sharp, fresh, clean citrus",
             "origin": "Sicily", "is_natural": True},

            {"name": "Grapefruit", "type": "citrus", "pyramid_level": "top",
             "volatility": 0.93, "strength": 0.65, "longevity": 0.18,
             "description": "Bitter-sweet, refreshing citrus",
             "origin": "USA", "is_natural": True},

            {"name": "Mandarin", "type": "citrus", "pyramid_level": "top",
             "volatility": 0.92, "strength": 0.5, "longevity": 0.2,
             "description": "Sweet, gentle citrus with floral notes",
             "origin": "China", "is_natural": True},

            {"name": "Black Pepper", "type": "spicy", "pyramid_level": "top",
             "volatility": 0.85, "strength": 0.8, "longevity": 0.3,
             "description": "Sharp, warm, spicy",
             "origin": "India", "is_natural": True},

            # Middle Notes (중간 휘발성)
            {"name": "Rose", "type": "floral", "pyramid_level": "middle",
             "volatility": 0.5, "strength": 0.8, "longevity": 0.6,
             "description": "Classic floral, romantic, powdery",
             "origin": "Bulgaria", "is_natural": True},

            {"name": "Jasmine", "type": "floral", "pyramid_level": "middle",
             "volatility": 0.45, "strength": 0.9, "longevity": 0.65,
             "description": "Intense, sweet, narcotic floral",
             "origin": "Egypt", "is_natural": True},

            {"name": "Lavender", "type": "aromatic", "pyramid_level": "middle",
             "volatility": 0.6, "strength": 0.7, "longevity": 0.5,
             "description": "Fresh, herbaceous, calming",
             "origin": "France", "is_natural": True},

            {"name": "Geranium", "type": "floral", "pyramid_level": "middle",
             "volatility": 0.55, "strength": 0.65, "longevity": 0.55,
             "description": "Rose-like with minty undertone",
             "origin": "Egypt", "is_natural": True},

            {"name": "Ylang-Ylang", "type": "floral", "pyramid_level": "middle",
             "volatility": 0.4, "strength": 0.85, "longevity": 0.7,
             "description": "Sweet, creamy, exotic floral",
             "origin": "Madagascar", "is_natural": True},

            # Base Notes (낮은 휘발성)
            {"name": "Sandalwood", "type": "woody", "pyramid_level": "base",
             "volatility": 0.1, "strength": 0.6, "longevity": 0.95,
             "description": "Creamy, soft, warm woody",
             "origin": "India", "is_natural": True},

            {"name": "Cedarwood", "type": "woody", "pyramid_level": "base",
             "volatility": 0.15, "strength": 0.65, "longevity": 0.9,
             "description": "Dry, clean, pencil shavings",
             "origin": "USA", "is_natural": True},

            {"name": "Patchouli", "type": "woody", "pyramid_level": "base",
             "volatility": 0.08, "strength": 0.9, "longevity": 0.98,
             "description": "Earthy, dark, mysterious",
             "origin": "Indonesia", "is_natural": True},

            {"name": "Vanilla", "type": "sweet", "pyramid_level": "base",
             "volatility": 0.2, "strength": 0.8, "longevity": 0.85,
             "description": "Sweet, warm, comforting",
             "origin": "Madagascar", "is_natural": True},

            {"name": "Musk", "type": "animalic", "pyramid_level": "base",
             "volatility": 0.05, "strength": 0.7, "longevity": 1.0,
             "description": "Clean, soft, skin-like",
             "origin": "Synthetic", "is_natural": False},

            {"name": "Amber", "type": "resinous", "pyramid_level": "base",
             "volatility": 0.12, "strength": 0.85, "longevity": 0.95,
             "description": "Warm, resinous, sweet",
             "origin": "Synthetic", "is_natural": False},

            {"name": "Vetiver", "type": "woody", "pyramid_level": "base",
             "volatility": 0.08, "strength": 0.75, "longevity": 0.97,
             "description": "Earthy, rooty, smoky",
             "origin": "Haiti", "is_natural": True},
        ]

        existing_notes = {n.name for n in session.query(Note.name).all()}

        for note_data in notes_data:
            if note_data["name"] not in existing_notes:
                # 텍스트로 임베딩 생성
                text = f"{note_data['name']} {note_data['type']} {note_data['description']}"
                embedding = self.encoder.encode(text)

                note = Note(
                    **note_data,
                    embedding=embedding.tolist()
                )
                session.add(note)

        session.flush()
        print(f"향료 노트 {len(notes_data)}개 로드 완료")

    def load_blending_rules(self, session):
        """블렌딩 규칙 로드"""
        rules = [
            # 조화로운 조합
            {"notes": ["Bergamot", "Lavender"], "compatibility": 0.9,
             "rule_type": "harmony", "description": "Classic fresh combination"},

            {"notes": ["Rose", "Sandalwood"], "compatibility": 0.85,
             "rule_type": "harmony", "description": "Romantic woody-floral"},

            {"notes": ["Jasmine", "Vanilla"], "compatibility": 0.8,
             "rule_type": "harmony", "description": "Sweet floral blend"},

            {"notes": ["Lemon", "Cedarwood"], "compatibility": 0.75,
             "rule_type": "harmony", "description": "Fresh woody citrus"},

            {"notes": ["Patchouli", "Vanilla"], "compatibility": 0.85,
             "rule_type": "enhancement", "description": "Oriental warmth"},

            # 충돌하는 조합
            {"notes": ["Lavender", "Patchouli"], "compatibility": -0.5,
             "rule_type": "conflict", "description": "Opposing characters"},

            {"notes": ["Lemon", "Vanilla"], "compatibility": -0.3,
             "rule_type": "conflict", "description": "Curdling effect"},
        ]

        # 노트 이름 -> ID 매핑
        note_map = {n.name: n.id for n in session.query(Note).all()}

        for rule in rules:
            if len(rule["notes"]) == 2:
                note1, note2 = rule["notes"]
                if note1 in note_map and note2 in note_map:
                    existing = session.query(BlendingRule).filter(
                        BlendingRule.note1_id == note_map[note1],
                        BlendingRule.note2_id == note_map[note2]
                    ).first()

                    if not existing:
                        blending_rule = BlendingRule(
                            note1_id=note_map[note1],
                            note2_id=note_map[note2],
                            compatibility=rule["compatibility"],
                            rule_type=rule["rule_type"],
                            description=rule["description"],
                            is_verified=True
                        )
                        session.add(blending_rule)

        session.flush()
        print(f"블렌딩 규칙 {len(rules)}개 로드 완료")

    def load_accord_templates(self, session):
        """어코드 템플릿 로드"""
        templates = [
            {
                "name": "Fougère Accord",
                "type": "fougere",
                "structure": {
                    "top": [{"note": "Lavender", "min": 10, "max": 20}],
                    "middle": [{"note": "Geranium", "min": 5, "max": 10}],
                    "base": [
                        {"note": "Cedarwood", "min": 10, "max": 15},
                        {"note": "Musk", "min": 5, "max": 10}
                    ]
                },
                "description": "Classic masculine accord with lavender and woody base"
            },
            {
                "name": "Chypre Accord",
                "type": "chypre",
                "structure": {
                    "top": [{"note": "Bergamot", "min": 15, "max": 25}],
                    "middle": [
                        {"note": "Rose", "min": 5, "max": 10},
                        {"note": "Jasmine", "min": 5, "max": 10}
                    ],
                    "base": [
                        {"note": "Patchouli", "min": 10, "max": 20},
                        {"note": "Musk", "min": 5, "max": 10}
                    ]
                },
                "description": "Sophisticated accord with citrus top and mossy base"
            },
            {
                "name": "Oriental Accord",
                "type": "oriental",
                "structure": {
                    "top": [{"note": "Mandarin", "min": 5, "max": 10}],
                    "middle": [
                        {"note": "Jasmine", "min": 10, "max": 15},
                        {"note": "Ylang-Ylang", "min": 5, "max": 10}
                    ],
                    "base": [
                        {"note": "Vanilla", "min": 15, "max": 25},
                        {"note": "Amber", "min": 10, "max": 20},
                        {"note": "Sandalwood", "min": 10, "max": 15}
                    ]
                },
                "description": "Warm, sweet, and sensual accord"
            }
        ]

        for template in templates:
            existing = session.query(AccordTemplate).filter_by(name=template["name"]).first()
            if not existing:
                # 임베딩 생성
                text = f"{template['name']} {template['type']} {template['description']}"
                embedding = self.encoder.encode(text)

                accord = AccordTemplate(
                    name=template["name"],
                    type=template["type"],
                    structure=template["structure"],
                    description=template["description"],
                    embedding=embedding.tolist()
                )
                session.add(accord)

        session.flush()
        print(f"어코드 템플릿 {len(templates)}개 로드 완료")

    def load_knowledge_base(self, session):
        """지식베이스 데이터 로드"""
        knowledge_entries = [
            {
                "category": "history",
                "title": "The Birth of Modern Perfumery",
                "content": "Modern perfumery began in the late 19th century with the synthesis of coumarin by William Perkin. This marked the beginning of using synthetic materials in perfume creation.",
                "tags": ["history", "synthetic", "innovation"],
                "confidence": 1.0
            },
            {
                "category": "technique",
                "title": "Perfume Pyramid Structure",
                "content": "Perfumes are structured in three layers: top notes (immediate impression, 5-15 minutes), middle/heart notes (main body, 20-60 minutes), and base notes (foundation, hours to days).",
                "tags": ["structure", "pyramid", "composition"],
                "confidence": 1.0
            },
            {
                "category": "ingredient",
                "title": "Natural vs Synthetic Materials",
                "content": "Modern perfumery uses both natural and synthetic materials. Synthetics provide consistency, affordability, and sometimes better performance than natural equivalents.",
                "tags": ["materials", "synthetic", "natural"],
                "confidence": 0.95
            },
            {
                "category": "technique",
                "title": "Fixatives in Perfumery",
                "content": "Fixatives are materials that slow down evaporation of volatile components, extending the life of the perfume. Common fixatives include musk, ambergris, and benzoin.",
                "tags": ["fixative", "longevity", "base notes"],
                "confidence": 1.0
            },
            {
                "category": "culture",
                "title": "Perfume in Different Cultures",
                "content": "Fragrance preferences vary by culture. Middle Eastern perfumes favor oud and rose, while Western perfumes often emphasize fresh and citrus notes.",
                "tags": ["culture", "preferences", "regional"],
                "confidence": 0.9
            }
        ]

        for entry in knowledge_entries:
            existing = session.query(KnowledgeBase).filter_by(title=entry["title"]).first()
            if not existing:
                # 임베딩 생성
                text = f"{entry['title']} {entry['content']}"
                embedding = self.encoder.encode(text)

                kb_entry = KnowledgeBase(
                    **entry,
                    embedding=embedding.tolist()
                )
                session.add(kb_entry)

        session.flush()
        print(f"지식베이스 항목 {len(knowledge_entries)}개 로드 완료")

    def load_sample_fragrances(self, session):
        """샘플 향수 데이터 로드"""
        fragrances = [
            {
                "name": "Spring Romance",
                "brand": "AI Parfums",
                "family": "floral",
                "gender": "feminine",
                "year": 2024,
                "description": "A romantic floral bouquet with fresh citrus opening",
                "compositions": [
                    {"note": "Bergamot", "percentage": 15, "level": "top"},
                    {"note": "Lemon", "percentage": 10, "level": "top"},
                    {"note": "Rose", "percentage": 20, "level": "middle"},
                    {"note": "Jasmine", "percentage": 15, "level": "middle"},
                    {"note": "Sandalwood", "percentage": 20, "level": "base"},
                    {"note": "Musk", "percentage": 10, "level": "base"},
                    {"note": "Vanilla", "percentage": 10, "level": "base"}
                ]
            },
            {
                "name": "Urban Edge",
                "brand": "AI Parfums",
                "family": "woody",
                "gender": "masculine",
                "year": 2024,
                "description": "Modern masculine fragrance with woody and spicy notes",
                "compositions": [
                    {"note": "Grapefruit", "percentage": 20, "level": "top"},
                    {"note": "Black Pepper", "percentage": 10, "level": "top"},
                    {"note": "Lavender", "percentage": 15, "level": "middle"},
                    {"note": "Geranium", "percentage": 10, "level": "middle"},
                    {"note": "Cedarwood", "percentage": 25, "level": "base"},
                    {"note": "Vetiver", "percentage": 15, "level": "base"},
                    {"note": "Amber", "percentage": 5, "level": "base"}
                ]
            }
        ]

        # 노트 이름 -> ID 매핑
        note_map = {n.name: n.id for n in session.query(Note).all()}

        for frag_data in fragrances:
            existing = session.query(Fragrance).filter_by(name=frag_data["name"]).first()
            if not existing:
                # 향수 임베딩 생성
                text = f"{frag_data['name']} {frag_data['family']} {frag_data['description']}"
                embedding = self.encoder.encode(text)

                compositions = frag_data.pop("compositions")

                fragrance = Fragrance(
                    **frag_data,
                    embedding=embedding.tolist()
                )
                session.add(fragrance)
                session.flush()

                # 조합 추가
                for comp in compositions:
                    if comp["note"] in note_map:
                        composition = FragranceComposition(
                            fragrance_id=fragrance.id,
                            note_id=note_map[comp["note"]],
                            percentage=comp["percentage"],
                            pyramid_level=comp["level"]
                        )
                        session.add(composition)

        session.flush()
        print(f"샘플 향수 {len(fragrances)}개 로드 완료")


def main():
    """메인 실행 함수"""
    initializer = DatabaseInitializer()
    initializer.initialize()


if __name__ == "__main__":
    main()