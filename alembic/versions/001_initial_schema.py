# 🗃️ 초기 데이터베이스 스키마 생성
"""Initial database schema

Revision ID: 001_initial_schema
Revises:
Create Date: 2024-01-20 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '001_initial_schema'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """초기 스키마 생성"""

    # 1. 사용자 테이블
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('username', sa.String(50), nullable=False),
        sa.Column('email', sa.String(255), nullable=False),
        sa.Column('password_hash', sa.String(255), nullable=False),
        sa.Column('full_name', sa.String(100), nullable=True),
        sa.Column('role', sa.Enum('user', 'admin', 'premium', name='user_role'), nullable=False, default='user'),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.Column('preferences', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('last_login', sa.TIMESTAMP(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('username'),
        sa.UniqueConstraint('email')
    )

    # 2. 향수 데이터 테이블
    op.create_table(
        'fragrances',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('brand', sa.String(100), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('fragrance_family', sa.String(50), nullable=True),
        sa.Column('intensity', sa.Enum('light', 'moderate', 'strong', name='intensity_level'), nullable=True),
        sa.Column('gender', sa.Enum('masculine', 'feminine', 'unisex', name='gender_type'), nullable=True),
        sa.Column('season', sa.ARRAY(sa.String(20)), nullable=True),
        sa.Column('occasion', sa.ARRAY(sa.String(50)), nullable=True),
        sa.Column('notes', sa.JSON(), nullable=False),  # {top: [], middle: [], base: []}
        sa.Column('price_range', sa.String(20), nullable=True),
        sa.Column('release_year', sa.Integer(), nullable=True),
        sa.Column('perfumer', sa.String(100), nullable=True),
        sa.Column('mood_tags', sa.ARRAY(sa.String(50)), nullable=True),
        sa.Column('longevity', sa.Integer(), nullable=True),  # hours
        sa.Column('projection', sa.Enum('intimate', 'moderate', 'strong', 'beast', name='projection_level'), nullable=True),
        sa.Column('rating', sa.Float(), nullable=True),
        sa.Column('rating_count', sa.Integer(), nullable=False, default=0),
        sa.Column('image_url', sa.String(500), nullable=True),
        sa.Column('external_id', sa.String(100), nullable=True),
        sa.Column('source', sa.String(50), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )

    # 3. 향수 임베딩 테이블
    op.create_table(
        'fragrance_embeddings',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('fragrance_id', sa.Integer(), nullable=False),
        sa.Column('embedding_model', sa.String(100), nullable=False),
        sa.Column('embedding_vector', postgresql.ARRAY(sa.Float), nullable=False),
        sa.Column('embedding_dimension', sa.Integer(), nullable=False),
        sa.Column('text_content', sa.Text(), nullable=False),
        sa.Column('embedding_type', sa.String(50), nullable=False),  # 'description', 'notes', 'combined'
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['fragrance_id'], ['fragrances.id'], ondelete='CASCADE')
    )

    # 4. 사용자 검색 이력
    op.create_table(
        'search_history',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('session_id', sa.String(100), nullable=True),
        sa.Column('query', sa.Text(), nullable=False),
        sa.Column('search_type', sa.String(50), nullable=False),
        sa.Column('filters', sa.JSON(), nullable=True),
        sa.Column('result_count', sa.Integer(), nullable=False),
        sa.Column('results', sa.JSON(), nullable=True),  # top results
        sa.Column('response_time_ms', sa.Float(), nullable=True),
        sa.Column('ip_address', sa.String(45), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='SET NULL')
    )

    # 5. 향수 생성 이력
    op.create_table(
        'generation_history',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('session_id', sa.String(100), nullable=True),
        sa.Column('prompt', sa.Text(), nullable=False),
        sa.Column('parameters', sa.JSON(), nullable=True),
        sa.Column('generated_recipe', sa.JSON(), nullable=False),
        sa.Column('model_name', sa.String(100), nullable=False),
        sa.Column('model_version', sa.String(50), nullable=True),
        sa.Column('generation_time_ms', sa.Float(), nullable=True),
        sa.Column('quality_score', sa.Float(), nullable=True),
        sa.Column('user_rating', sa.Integer(), nullable=True),  # 1-5
        sa.Column('is_favorite', sa.Boolean(), nullable=False, default=False),
        sa.Column('ip_address', sa.String(45), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='SET NULL')
    )

    # 6. 사용자 향수 컬렉션
    op.create_table(
        'user_collections',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('fragrance_id', sa.Integer(), nullable=False),
        sa.Column('collection_type', sa.Enum('owned', 'wishlist', 'tried', 'disliked', name='collection_type'), nullable=False),
        sa.Column('rating', sa.Integer(), nullable=True),  # 1-5
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('purchase_date', sa.Date(), nullable=True),
        sa.Column('purchase_price', sa.Float(), nullable=True),
        sa.Column('size_ml', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['fragrance_id'], ['fragrances.id'], ondelete='CASCADE'),
        sa.UniqueConstraint('user_id', 'fragrance_id', 'collection_type')
    )

    # 7. 향수 리뷰
    op.create_table(
        'fragrance_reviews',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('fragrance_id', sa.Integer(), nullable=False),
        sa.Column('rating', sa.Integer(), nullable=False),  # 1-5
        sa.Column('title', sa.String(200), nullable=True),
        sa.Column('content', sa.Text(), nullable=True),
        sa.Column('longevity_rating', sa.Integer(), nullable=True),  # 1-5
        sa.Column('projection_rating', sa.Integer(), nullable=True),  # 1-5
        sa.Column('value_rating', sa.Integer(), nullable=True),  # 1-5
        sa.Column('occasion_tags', sa.ARRAY(sa.String(50)), nullable=True),
        sa.Column('season_tags', sa.ARRAY(sa.String(20)), nullable=True),
        sa.Column('is_verified_purchase', sa.Boolean(), nullable=False, default=False),
        sa.Column('helpful_count', sa.Integer(), nullable=False, default=0),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['fragrance_id'], ['fragrances.id'], ondelete='CASCADE'),
        sa.UniqueConstraint('user_id', 'fragrance_id')
    )

    # 8. API 사용 통계
    op.create_table(
        'api_usage_stats',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('endpoint', sa.String(200), nullable=False),
        sa.Column('method', sa.String(10), nullable=False),
        sa.Column('status_code', sa.Integer(), nullable=False),
        sa.Column('response_time_ms', sa.Float(), nullable=True),
        sa.Column('request_size_bytes', sa.Integer(), nullable=True),
        sa.Column('response_size_bytes', sa.Integer(), nullable=True),
        sa.Column('ip_address', sa.String(45), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('api_key_id', sa.String(100), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='SET NULL')
    )

    # 9. 시스템 설정
    op.create_table(
        'system_settings',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('key', sa.String(100), nullable=False),
        sa.Column('value', sa.JSON(), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('key')
    )

    # 10. 모델 메타데이터
    op.create_table(
        'model_metadata',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('model_name', sa.String(100), nullable=False),
        sa.Column('model_type', sa.String(50), nullable=False),  # 'embedding', 'generation'
        sa.Column('model_version', sa.String(50), nullable=False),
        sa.Column('model_path', sa.String(500), nullable=True),
        sa.Column('parameters', sa.JSON(), nullable=True),
        sa.Column('training_data', sa.JSON(), nullable=True),
        sa.Column('performance_metrics', sa.JSON(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('model_name', 'model_version')
    )

    # 인덱스 생성
    create_indexes()

    # 기본 데이터 삽입
    insert_default_data()


def create_indexes() -> None:
    """성능 최적화를 위한 인덱스 생성"""

    # 사용자 관련 인덱스
    op.create_index('idx_users_email', 'users', ['email'])
    op.create_index('idx_users_username', 'users', ['username'])
    op.create_index('idx_users_created_at', 'users', ['created_at'])

    # 향수 관련 인덱스
    op.create_index('idx_fragrances_name', 'fragrances', ['name'])
    op.create_index('idx_fragrances_brand', 'fragrances', ['brand'])
    op.create_index('idx_fragrances_family', 'fragrances', ['fragrance_family'])
    op.create_index('idx_fragrances_rating', 'fragrances', ['rating'])
    op.create_index('idx_fragrances_active', 'fragrances', ['is_active'])
    op.create_index('idx_fragrances_created_at', 'fragrances', ['created_at'])

    # 임베딩 관련 인덱스
    op.create_index('idx_embeddings_fragrance_id', 'fragrance_embeddings', ['fragrance_id'])
    op.create_index('idx_embeddings_model', 'fragrance_embeddings', ['embedding_model'])
    op.create_index('idx_embeddings_type', 'fragrance_embeddings', ['embedding_type'])

    # 검색 이력 인덱스
    op.create_index('idx_search_history_user_id', 'search_history', ['user_id'])
    op.create_index('idx_search_history_created_at', 'search_history', ['created_at'])
    op.create_index('idx_search_history_session_id', 'search_history', ['session_id'])

    # 생성 이력 인덱스
    op.create_index('idx_generation_history_user_id', 'generation_history', ['user_id'])
    op.create_index('idx_generation_history_created_at', 'generation_history', ['created_at'])
    op.create_index('idx_generation_history_favorite', 'generation_history', ['is_favorite'])

    # 컬렉션 인덱스
    op.create_index('idx_user_collections_user_id', 'user_collections', ['user_id'])
    op.create_index('idx_user_collections_fragrance_id', 'user_collections', ['fragrance_id'])
    op.create_index('idx_user_collections_type', 'user_collections', ['collection_type'])

    # 리뷰 인덱스
    op.create_index('idx_reviews_fragrance_id', 'fragrance_reviews', ['fragrance_id'])
    op.create_index('idx_reviews_user_id', 'fragrance_reviews', ['user_id'])
    op.create_index('idx_reviews_rating', 'fragrance_reviews', ['rating'])
    op.create_index('idx_reviews_active', 'fragrance_reviews', ['is_active'])

    # API 사용 통계 인덱스
    op.create_index('idx_api_stats_endpoint', 'api_usage_stats', ['endpoint'])
    op.create_index('idx_api_stats_created_at', 'api_usage_stats', ['created_at'])
    op.create_index('idx_api_stats_user_id', 'api_usage_stats', ['user_id'])
    op.create_index('idx_api_stats_status_code', 'api_usage_stats', ['status_code'])


def insert_default_data() -> None:
    """기본 데이터 삽입"""
    from sqlalchemy import text

    # 기본 시스템 설정
    default_settings = [
        {
            'key': 'search_default_top_k',
            'value': '{"value": 10, "min": 1, "max": 100}',
            'description': '기본 검색 결과 수'
        },
        {
            'key': 'generation_default_temperature',
            'value': '{"value": 0.7, "min": 0.1, "max": 1.0}',
            'description': '기본 생성 온도'
        },
        {
            'key': 'max_requests_per_minute',
            'value': '{"user": 60, "premium": 200, "admin": 1000}',
            'description': '분당 최대 요청 수'
        },
        {
            'key': 'supported_languages',
            'value': '["ko", "en", "ja", "zh", "es", "fr"]',
            'description': '지원 언어 목록'
        },
        {
            'key': 'cache_ttl_seconds',
            'value': '{"search": 300, "generation": 600, "user_data": 1800}',
            'description': '캐시 만료 시간 (초)'
        }
    ]

    for setting in default_settings:
        op.execute(text("""
            INSERT INTO system_settings (key, value, description)
            VALUES (:key, :value::json, :description)
        """), setting)

    # 기본 관리자 계정 (비밀번호: admin123 - 반드시 변경 필요)
    op.execute(text("""
        INSERT INTO users (username, email, password_hash, full_name, role, is_active)
        VALUES (
            'admin',
            'admin@fragrance-ai.com',
            '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewBTKhUCKi1T1QBe',
            'System Administrator',
            'admin',
            true
        )
    """))


def downgrade() -> None:
    """스키마 다운그레이드"""
    # 인덱스 삭제
    op.drop_index('idx_api_stats_status_code')
    op.drop_index('idx_api_stats_user_id')
    op.drop_index('idx_api_stats_created_at')
    op.drop_index('idx_api_stats_endpoint')

    op.drop_index('idx_reviews_active')
    op.drop_index('idx_reviews_rating')
    op.drop_index('idx_reviews_user_id')
    op.drop_index('idx_reviews_fragrance_id')

    op.drop_index('idx_user_collections_type')
    op.drop_index('idx_user_collections_fragrance_id')
    op.drop_index('idx_user_collections_user_id')

    op.drop_index('idx_generation_history_favorite')
    op.drop_index('idx_generation_history_created_at')
    op.drop_index('idx_generation_history_user_id')

    op.drop_index('idx_search_history_session_id')
    op.drop_index('idx_search_history_created_at')
    op.drop_index('idx_search_history_user_id')

    op.drop_index('idx_embeddings_type')
    op.drop_index('idx_embeddings_model')
    op.drop_index('idx_embeddings_fragrance_id')

    op.drop_index('idx_fragrances_created_at')
    op.drop_index('idx_fragrances_active')
    op.drop_index('idx_fragrances_rating')
    op.drop_index('idx_fragrances_family')
    op.drop_index('idx_fragrances_brand')
    op.drop_index('idx_fragrances_name')

    op.drop_index('idx_users_created_at')
    op.drop_index('idx_users_username')
    op.drop_index('idx_users_email')

    # 테이블 삭제
    op.drop_table('model_metadata')
    op.drop_table('system_settings')
    op.drop_table('api_usage_stats')
    op.drop_table('fragrance_reviews')
    op.drop_table('user_collections')
    op.drop_table('generation_history')
    op.drop_table('search_history')
    op.drop_table('fragrance_embeddings')
    op.drop_table('fragrances')
    op.drop_table('users')

    # Enum 타입 삭제
    op.execute('DROP TYPE IF EXISTS projection_level CASCADE')
    op.execute('DROP TYPE IF EXISTS collection_type CASCADE')
    op.execute('DROP TYPE IF EXISTS gender_type CASCADE')
    op.execute('DROP TYPE IF EXISTS intensity_level CASCADE')
    op.execute('DROP TYPE IF EXISTS user_role CASCADE')