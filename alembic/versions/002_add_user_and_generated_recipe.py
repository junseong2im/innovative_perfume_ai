"""add user and generated recipe tables

Revision ID: 002
Revises: 001
Create Date: 2025-09-24 09:50:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None


def upgrade():
    # Create users table
    op.create_table('users',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('username', sa.String(length=100), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('full_name', sa.String(length=200), nullable=True),
        sa.Column('phone', sa.String(length=50), nullable=True),
        sa.Column('profile_image_url', sa.String(length=500), nullable=True),
        sa.Column('hashed_password', sa.String(length=255), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('is_verified', sa.Boolean(), nullable=True),
        sa.Column('role', sa.String(length=50), nullable=True),
        sa.Column('preferences', sa.JSON(), nullable=True),
        sa.Column('favorite_notes', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.CheckConstraint("role IN ('customer', 'admin', 'expert')", name='check_user_role'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)
    op.create_index(op.f('ix_users_username'), 'users', ['username'], unique=True)

    # Create generated_recipes table
    op.create_table('generated_recipes',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('user_id', sa.String(), nullable=False),
        sa.Column('conversation_id', sa.String(length=100), nullable=False),
        sa.Column('user_prompt', sa.Text(), nullable=False),
        sa.Column('conversation_history', sa.JSON(), nullable=True),
        sa.Column('recipe_name', sa.String(length=200), nullable=False),
        sa.Column('recipe_description', sa.Text(), nullable=True),
        sa.Column('master_formula', sa.JSON(), nullable=False),
        sa.Column('top_notes', sa.JSON(), nullable=True),
        sa.Column('heart_notes', sa.JSON(), nullable=True),
        sa.Column('base_notes', sa.JSON(), nullable=True),
        sa.Column('harmony_score', sa.Float(), nullable=True),
        sa.Column('stability_score', sa.Float(), nullable=True),
        sa.Column('longevity_score', sa.Float(), nullable=True),
        sa.Column('sillage_score', sa.Float(), nullable=True),
        sa.Column('overall_score', sa.Float(), nullable=True),
        sa.Column('generation_model', sa.String(length=100), nullable=True),
        sa.Column('validation_model', sa.String(length=100), nullable=True),
        sa.Column('generation_timestamp', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_validated', sa.Boolean(), nullable=True),
        sa.Column('is_approved', sa.Boolean(), nullable=True),
        sa.Column('admin_notes', sa.Text(), nullable=True),
        sa.Column('user_rating', sa.Float(), nullable=True),
        sa.Column('user_feedback', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.CheckConstraint('harmony_score >= 0 AND harmony_score <= 10', name='check_harmony_range'),
        sa.CheckConstraint('longevity_score >= 0 AND longevity_score <= 10', name='check_longevity_range'),
        sa.CheckConstraint('overall_score >= 0 AND overall_score <= 10', name='check_overall_range'),
        sa.CheckConstraint('sillage_score >= 0 AND sillage_score <= 10', name='check_sillage_range'),
        sa.CheckConstraint('stability_score >= 0 AND stability_score <= 10', name='check_stability_range'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_generated_recipes_conversation'), 'generated_recipes', ['conversation_id'], unique=False)
    op.create_index(op.f('ix_generated_recipes_timestamp'), 'generated_recipes', ['generation_timestamp'], unique=False)
    op.create_index(op.f('ix_generated_recipes_user'), 'generated_recipes', ['user_id'], unique=False)


def downgrade():
    # Drop generated_recipes table
    op.drop_index(op.f('ix_generated_recipes_user'), table_name='generated_recipes')
    op.drop_index(op.f('ix_generated_recipes_timestamp'), table_name='generated_recipes')
    op.drop_index(op.f('ix_generated_recipes_conversation'), table_name='generated_recipes')
    op.drop_table('generated_recipes')

    # Drop users table
    op.drop_index(op.f('ix_users_username'), table_name='users')
    op.drop_index(op.f('ix_users_email'), table_name='users')
    op.drop_table('users')