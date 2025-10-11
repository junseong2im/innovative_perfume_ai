# fragrance_ai/services/__init__.py
"""
Service layer for fragrance AI
"""

from .evolution_service import EvolutionService, get_evolution_service

__all__ = [
    'EvolutionService',
    'get_evolution_service'
]