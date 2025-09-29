"""
Living Scent Database Models
향수 DNA와 표현형을 저장하는 데이터베이스 모델
"""

from sqlalchemy import Column, String, Float, JSON, DateTime, Integer, ForeignKey, Text, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .base import Base
import uuid


class OlfactoryDNAModel(Base):
    """향수 DNA 데이터베이스 모델"""
    __tablename__ = "olfactory_dna"

    # Primary key
    dna_id = Column(String(50), primary_key=True, default=lambda: f"DNA_{uuid.uuid4().hex[:12].upper()}")

    # DNA 정보
    lineage = Column(JSON, nullable=False)  # 부모 계보
    genotype = Column(JSON, nullable=False)  # 유전자형 (top, middle, base notes)
    phenotype_potential = Column(JSON, nullable=False)  # 표현형 잠재력

    # 메타데이터
    story = Column(Text, nullable=False)  # DNA 스토리
    generation = Column(Integer, default=1)  # 세대 수
    fitness_score = Column(Float, default=0.0)  # 적합도 점수
    mutation_history = Column(JSON, default=list)  # 돌연변이 이력

    # 생성 정보
    created_by_user_id = Column(String(100), nullable=True)  # 생성한 사용자 ID
    creation_context = Column(JSON, nullable=True)  # 생성 컨텍스트 (원본 요청 등)

    # 통계
    total_phenotypes = Column(Integer, default=0)  # 파생된 표현형 수
    average_rating = Column(Float, default=0.0)  # 평균 평점
    usage_count = Column(Integer, default=0)  # 사용 횟수

    # 타임스탬프
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # 활성 상태
    is_active = Column(Boolean, default=True)
    is_public = Column(Boolean, default=True)  # 공개 여부

    # 관계
    phenotypes = relationship("ScentPhenotypeModel", back_populates="dna", cascade="all, delete-orphan")
    favorites = relationship("UserFavoriteModel", back_populates="dna", cascade="all, delete-orphan")


class ScentPhenotypeModel(Base):
    """향수 표현형 데이터베이스 모델"""
    __tablename__ = "scent_phenotypes"

    # Primary key
    phenotype_id = Column(String(50), primary_key=True, default=lambda: f"PHENO_{uuid.uuid4().hex[:10].upper()}")

    # DNA 연결
    based_on_dna = Column(String(50), ForeignKey("olfactory_dna.dna_id", ondelete="CASCADE"), nullable=False)

    # 표현형 정보
    epigenetic_trigger = Column(Text, nullable=False)  # 후생유전학적 유발 요인
    recipe = Column(JSON, nullable=False)  # 실제 레시피
    modifications = Column(JSON, nullable=False)  # 적용된 후생유전학적 수정
    description = Column(Text, nullable=False)  # 표현형 설명
    environmental_response = Column(JSON, nullable=False)  # 환경 반응성

    # 진화 추적
    parent_phenotype = Column(String(50), nullable=True)  # 부모 표현형 ID
    evolution_path = Column(JSON, default=list)  # 진화 경로
    evolution_generation = Column(Integer, default=1)  # 진화 세대

    # 사용자 정보
    created_by_user_id = Column(String(100), nullable=True)  # 생성한 사용자 ID
    feedback_context = Column(JSON, nullable=True)  # 피드백 컨텍스트

    # 평가
    user_rating = Column(Float, nullable=True)  # 사용자 평점 (1-5)
    expert_rating = Column(Float, nullable=True)  # 전문가 평점
    harmony_score = Column(Float, nullable=True)  # 조화도 점수
    uniqueness_score = Column(Float, nullable=True)  # 독창성 점수

    # 타임스탬프
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # 활성 상태
    is_active = Column(Boolean, default=True)
    is_featured = Column(Boolean, default=False)  # 추천 여부

    # 관계
    dna = relationship("OlfactoryDNAModel", back_populates="phenotypes")
    user_interactions = relationship("UserInteractionModel", back_populates="phenotype", cascade="all, delete-orphan")


class UserInteractionModel(Base):
    """사용자 상호작용 모델"""
    __tablename__ = "user_interactions"

    # Primary key
    interaction_id = Column(String(50), primary_key=True, default=lambda: uuid.uuid4().hex)

    # 연결
    user_id = Column(String(100), nullable=False)
    phenotype_id = Column(String(50), ForeignKey("scent_phenotypes.phenotype_id", ondelete="CASCADE"), nullable=True)
    dna_id = Column(String(50), ForeignKey("olfactory_dna.dna_id", ondelete="CASCADE"), nullable=True)

    # 상호작용 정보
    interaction_type = Column(String(50), nullable=False)  # 'create', 'evolve', 'rate', 'favorite', 'share'
    interaction_data = Column(JSON, nullable=True)  # 상호작용 상세 데이터

    # 피드백
    feedback_text = Column(Text, nullable=True)
    satisfaction_score = Column(Float, nullable=True)

    # 타임스탬프
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # 관계
    phenotype = relationship("ScentPhenotypeModel", back_populates="user_interactions")


class UserFavoriteModel(Base):
    """사용자 즐겨찾기 모델"""
    __tablename__ = "user_favorites"

    # Primary key
    favorite_id = Column(String(50), primary_key=True, default=lambda: uuid.uuid4().hex)

    # 연결
    user_id = Column(String(100), nullable=False)
    dna_id = Column(String(50), ForeignKey("olfactory_dna.dna_id", ondelete="CASCADE"), nullable=True)

    # 메타데이터
    notes = Column(Text, nullable=True)  # 사용자 노트
    tags = Column(JSON, default=list)  # 사용자 태그

    # 타임스탬프
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # 관계
    dna = relationship("OlfactoryDNAModel", back_populates="favorites")


class FragranceEvolutionTreeModel(Base):
    """향수 진화 트리 모델 - DNA와 표현형의 계보 추적"""
    __tablename__ = "fragrance_evolution_trees"

    # Primary key
    tree_id = Column(String(50), primary_key=True, default=lambda: uuid.uuid4().hex)

    # 루트 DNA
    root_dna_id = Column(String(50), ForeignKey("olfactory_dna.dna_id"), nullable=False)

    # 트리 정보
    tree_name = Column(String(200), nullable=True)
    tree_description = Column(Text, nullable=True)
    total_nodes = Column(Integer, default=1)  # 총 노드 수
    max_depth = Column(Integer, default=0)  # 최대 깊이

    # 진화 통계
    total_mutations = Column(Integer, default=0)  # 총 돌연변이 수
    total_epigenetic_variations = Column(Integer, default=0)  # 총 후생유전학적 변이 수

    # 트리 구조 (JSON으로 저장)
    tree_structure = Column(JSON, nullable=False)  # 전체 트리 구조

    # 생성 정보
    created_by_user_id = Column(String(100), nullable=True)

    # 타임스탬프
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # 활성 상태
    is_active = Column(Boolean, default=True)
    is_public = Column(Boolean, default=True)