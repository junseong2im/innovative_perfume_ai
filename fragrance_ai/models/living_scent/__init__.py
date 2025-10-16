"""
Living Scent Models Package
살아있는 향수 AI 모델 패키지
"""

from .linguistic_receptor import (
    LinguisticReceptorAI,
    get_linguistic_receptor,
    UserIntent,
    StructuredInput
)

from .cognitive_core import (
    CognitiveCoreAI,
    get_cognitive_core,
    CreativeBrief
)

from .olfactory_recombinator import (
    OlfactoryRecombinatorAI,
    get_olfactory_recombinator,
    OlfactoryDNA,
    FragranceGene
)

from .epigenetic_variation import (
    EpigeneticVariationAI,
    get_epigenetic_variation,
    ScentPhenotype,
    EpigeneticMarker,
    EpigeneticModification
)

__all__ = [
    # Linguistic Receptor
    'LinguisticReceptorAI',
    'get_linguistic_receptor',
    'UserIntent',
    'StructuredInput',

    # Cognitive Core
    'CognitiveCoreAI',
    'get_cognitive_core',
    'CreativeBrief',

    # Olfactory Recombinator
    'OlfactoryRecombinatorAI',
    'get_olfactory_recombinator',
    'OlfactoryDNA',
    'FragranceGene',

    # Epigenetic Variation
    'EpigeneticVariationAI',
    'get_epigenetic_variation',
    'ScentPhenotype',
    'EpigeneticMarker',
    'EpigeneticModification'
]

# 버전 정보
__version__ = '1.0.0'
__author__ = 'Living Scent AI Team'