"""
Central database base configuration
모든 SQLAlchemy 모델이 사용하는 단일 Base 클래스
"""

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import MetaData

# Naming convention for constraints
convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s"
}

# Create metadata with naming convention
metadata = MetaData(naming_convention=convention)

# Single Base instance for all models
Base = declarative_base(metadata=metadata)

# Database session getter
def get_db():
    """Get database session (placeholder for compatibility)"""
    from sqlalchemy.orm import Session
    # This is a placeholder function for compatibility
    # In production, use DatabaseConnectionManager
    return None

# Export
__all__ = ['Base', 'metadata', 'get_db']