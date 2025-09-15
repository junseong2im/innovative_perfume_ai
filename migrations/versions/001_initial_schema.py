"""Initial database schema

Revision ID: 001_initial_schema
Revises:
Create Date: 2025-09-15 12:00:00.000000

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
    # Create users table
    op.create_table('users',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('username', sa.String(length=50), nullable=False),
    sa.Column('email', sa.String(length=255), nullable=False),
    sa.Column('hashed_password', sa.String(length=255), nullable=False),
    sa.Column('is_active', sa.Boolean(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('email'),
    sa.UniqueConstraint('username')
    )
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=False)
    op.create_index(op.f('ix_users_id'), 'users', ['id'], unique=False)
    op.create_index(op.f('ix_users_username'), 'users', ['username'], unique=False)

    # Create fragrance_ingredients table
    op.create_table('fragrance_ingredients',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('name', sa.String(length=255), nullable=False),
    sa.Column('english_name', sa.String(length=255), nullable=True),
    sa.Column('korean_name', sa.String(length=255), nullable=True),
    sa.Column('category', sa.String(length=100), nullable=True),
    sa.Column('fragrance_family', sa.String(length=100), nullable=True),
    sa.Column('note_type', sa.String(length=50), nullable=True),
    sa.Column('description', sa.Text(), nullable=True),
    sa.Column('origin', sa.String(length=255), nullable=True),
    sa.Column('cas_number', sa.String(length=20), nullable=True),
    sa.Column('intensity', sa.Float(), nullable=True),
    sa.Column('longevity', sa.Float(), nullable=True),
    sa.Column('sillage', sa.Float(), nullable=True),
    sa.Column('price_range', sa.String(length=50), nullable=True),
    sa.Column('safety_rating', sa.String(length=10), nullable=True),
    sa.Column('allergen_info', sa.Text(), nullable=True),
    sa.Column('blending_guidelines', sa.Text(), nullable=True),
    sa.Column('supplier_info', sa.Text(), nullable=True),
    sa.Column('molecular_formula', sa.String(length=100), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('cas_number'),
    sa.UniqueConstraint('name')
    )
    op.create_index(op.f('ix_fragrance_ingredients_category'), 'fragrance_ingredients', ['category'], unique=False)
    op.create_index(op.f('ix_fragrance_ingredients_fragrance_family'), 'fragrance_ingredients', ['fragrance_family'], unique=False)
    op.create_index(op.f('ix_fragrance_ingredients_id'), 'fragrance_ingredients', ['id'], unique=False)
    op.create_index(op.f('ix_fragrance_ingredients_name'), 'fragrance_ingredients', ['name'], unique=False)
    op.create_index(op.f('ix_fragrance_ingredients_note_type'), 'fragrance_ingredients', ['note_type'], unique=False)

    # Create fragrance_recipes table
    op.create_table('fragrance_recipes',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('name', sa.String(length=255), nullable=False),
    sa.Column('description', sa.Text(), nullable=True),
    sa.Column('fragrance_family', sa.String(length=100), nullable=True),
    sa.Column('mood', sa.String(length=255), nullable=True),
    sa.Column('season', sa.String(length=100), nullable=True),
    sa.Column('gender', sa.String(length=50), nullable=True),
    sa.Column('intensity', sa.String(length=50), nullable=True),
    sa.Column('longevity', sa.String(length=50), nullable=True),
    sa.Column('sillage', sa.String(length=50), nullable=True),
    sa.Column('recipe_type', sa.String(length=50), nullable=True),
    sa.Column('total_percentage', sa.Float(), nullable=True),
    sa.Column('notes', sa.JSON(), nullable=True),
    sa.Column('instructions', sa.Text(), nullable=True),
    sa.Column('story', sa.Text(), nullable=True),
    sa.Column('tags', sa.JSON(), nullable=True),
    sa.Column('quality_score', sa.Float(), nullable=True),
    sa.Column('complexity_score', sa.Float(), nullable=True),
    sa.Column('creativity_score', sa.Float(), nullable=True),
    sa.Column('feasibility_score', sa.Float(), nullable=True),
    sa.Column('user_id', sa.Integer(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_fragrance_recipes_fragrance_family'), 'fragrance_recipes', ['fragrance_family'], unique=False)
    op.create_index(op.f('ix_fragrance_recipes_id'), 'fragrance_recipes', ['id'], unique=False)
    op.create_index(op.f('ix_fragrance_recipes_name'), 'fragrance_recipes', ['name'], unique=False)

    # Create recipe_ingredients table
    op.create_table('recipe_ingredients',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('recipe_id', sa.Integer(), nullable=False),
    sa.Column('ingredient_id', sa.Integer(), nullable=False),
    sa.Column('percentage', sa.Float(), nullable=False),
    sa.Column('note_type', sa.String(length=50), nullable=True),
    sa.Column('role', sa.String(length=100), nullable=True),
    sa.Column('notes', sa.Text(), nullable=True),
    sa.ForeignKeyConstraint(['ingredient_id'], ['fragrance_ingredients.id'], ),
    sa.ForeignKeyConstraint(['recipe_id'], ['fragrance_recipes.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('recipe_id', 'ingredient_id', name='uq_recipe_ingredient')
    )
    op.create_index(op.f('ix_recipe_ingredients_id'), 'recipe_ingredients', ['id'], unique=False)

    # Create user_preferences table
    op.create_table('user_preferences',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('preferred_families', sa.JSON(), nullable=True),
    sa.Column('disliked_ingredients', sa.JSON(), nullable=True),
    sa.Column('intensity_preference', sa.String(length=50), nullable=True),
    sa.Column('season_preferences', sa.JSON(), nullable=True),
    sa.Column('mood_preferences', sa.JSON(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('user_id')
    )
    op.create_index(op.f('ix_user_preferences_id'), 'user_preferences', ['id'], unique=False)

    # Create search_logs table
    op.create_table('search_logs',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=True),
    sa.Column('query', sa.String(length=1000), nullable=False),
    sa.Column('search_type', sa.String(length=50), nullable=True),
    sa.Column('results_count', sa.Integer(), nullable=True),
    sa.Column('response_time', sa.Float(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_search_logs_id'), 'search_logs', ['id'], unique=False)

    # Create generation_logs table
    op.create_table('generation_logs',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=True),
    sa.Column('prompt', sa.Text(), nullable=False),
    sa.Column('recipe_type', sa.String(length=50), nullable=True),
    sa.Column('quality_score', sa.Float(), nullable=True),
    sa.Column('response_time', sa.Float(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_generation_logs_id'), 'generation_logs', ['id'], unique=False)

    # Create model_metrics table
    op.create_table('model_metrics',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('model_name', sa.String(length=255), nullable=False),
    sa.Column('model_version', sa.String(length=50), nullable=True),
    sa.Column('metric_name', sa.String(length=100), nullable=False),
    sa.Column('metric_value', sa.Float(), nullable=False),
    sa.Column('evaluation_data', sa.JSON(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_model_metrics_id'), 'model_metrics', ['id'], unique=False)
    op.create_index(op.f('ix_model_metrics_model_name'), 'model_metrics', ['model_name'], unique=False)


def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_index(op.f('ix_model_metrics_model_name'), table_name='model_metrics')
    op.drop_index(op.f('ix_model_metrics_id'), table_name='model_metrics')
    op.drop_table('model_metrics')

    op.drop_index(op.f('ix_generation_logs_id'), table_name='generation_logs')
    op.drop_table('generation_logs')

    op.drop_index(op.f('ix_search_logs_id'), table_name='search_logs')
    op.drop_table('search_logs')

    op.drop_index(op.f('ix_user_preferences_id'), table_name='user_preferences')
    op.drop_table('user_preferences')

    op.drop_index(op.f('ix_recipe_ingredients_id'), table_name='recipe_ingredients')
    op.drop_table('recipe_ingredients')

    op.drop_index(op.f('ix_fragrance_recipes_name'), table_name='fragrance_recipes')
    op.drop_index(op.f('ix_fragrance_recipes_id'), table_name='fragrance_recipes')
    op.drop_index(op.f('ix_fragrance_recipes_fragrance_family'), table_name='fragrance_recipes')
    op.drop_table('fragrance_recipes')

    op.drop_index(op.f('ix_fragrance_ingredients_note_type'), table_name='fragrance_ingredients')
    op.drop_index(op.f('ix_fragrance_ingredients_name'), table_name='fragrance_ingredients')
    op.drop_index(op.f('ix_fragrance_ingredients_id'), table_name='fragrance_ingredients')
    op.drop_index(op.f('ix_fragrance_ingredients_fragrance_family'), table_name='fragrance_ingredients')
    op.drop_index(op.f('ix_fragrance_ingredients_category'), table_name='fragrance_ingredients')
    op.drop_table('fragrance_ingredients')

    op.drop_index(op.f('ix_users_username'), table_name='users')
    op.drop_index(op.f('ix_users_id'), table_name='users')
    op.drop_index(op.f('ix_users_email'), table_name='users')
    op.drop_table('users')