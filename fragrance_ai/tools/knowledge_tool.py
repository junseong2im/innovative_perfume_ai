"""
향수 지식 베이스 도구
- 향수 역사, 노트 정보, 제조 기법 등 도메인 지식 제공
- RAG (Retrieval-Augmented Generation) 지원
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import logging
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Pydantic 스키마
class KnowledgeQuery(BaseModel):
    """지식 쿼리"""
    category: str = Field(..., description="지식 카테고리: history, technique, note, accord, perfumer")
    query: str = Field(..., description="구체적인 질문")
    context: Optional[Dict[str, Any]] = Field(default=None, description="추가 컨텍스트")

class KnowledgeResponse(BaseModel):
    """지식 응답"""
    category: str
    query: str
    answer: str
    confidence: float = Field(default=0.0, description="응답 신뢰도")
    sources: List[str] = Field(default_factory=list, description="정보 출처")
    related_topics: List[str] = Field(default_factory=list, description="관련 주제")

class PerfumeKnowledgeBase:
    """향수 도메인 지식 베이스"""

    def __init__(self):
        """지식 베이스 초기화"""
        self.knowledge_data = self._load_knowledge_base()
        self.note_database = self._load_note_database()
        self.accord_formulas = self._load_accord_formulas()
        self.perfumer_styles = self._load_perfumer_styles()

    def _load_knowledge_base(self) -> dict:
        """지식 베이스 로드"""
        knowledge = {
            "history": {
                "ancient": {
                    "egypt": "고대 이집트는 향수의 발상지로, 종교 의식과 미라 제작에 향료를 사용했습니다.",
                    "greece": "고대 그리스에서는 향수를 '신들의 선물'로 여기며 올림픽 우승자에게 선물했습니다.",
                    "rome": "로마 제국은 향수 문화를 대중화시켰으며, 공중목욕탕에서 향유를 사용했습니다."
                },
                "medieval": {
                    "arabic": "아랍 문화권에서 증류 기술을 발전시켜 알코올 기반 향수를 개발했습니다.",
                    "europe": "중세 유럽에서는 향수를 질병 예방과 악취 제거용으로 사용했습니다."
                },
                "modern": {
                    "grasse": "프랑스 그라스는 17세기부터 세계 향수의 수도로 자리잡았습니다.",
                    "synthetic": "19세기 후반 합성 향료 개발로 현대 향수 산업이 시작되었습니다.",
                    "designer": "20세기에는 패션 하우스들이 시그니처 향수를 출시하기 시작했습니다."
                }
            },
            "techniques": {
                "extraction": {
                    "distillation": "증류법: 스팀을 이용해 휘발성 향 성분을 추출하는 가장 일반적인 방법",
                    "enfleurage": "앙플뢰라주: 동물성 지방에 꽃잎을 올려 향을 흡수시키는 전통 기법",
                    "expression": "압착법: 감귤류 껍질을 압착하여 에센셜 오일을 추출",
                    "solvent_extraction": "용매 추출: 화학 용매를 사용해 향 성분을 추출 (앱솔루트 제조)",
                    "co2_extraction": "초임계 CO2 추출: 현대적 방법으로 순수한 향 성분 추출"
                },
                "blending": {
                    "pyramid": "향수 피라미드: 탑-미들-베이스 노트의 3단 구조",
                    "accord": "어코드: 여러 향료를 조합해 새로운 향을 만드는 기술",
                    "modification": "모디피케이션: 기존 향수를 변형하여 새로운 버전 개발",
                    "layering": "레이어링: 여러 향을 겹쳐 복잡한 향 프로필 생성"
                },
                "maturation": {
                    "maceration": "침용: 알코올에 향료를 녹여 숙성시키는 과정",
                    "aging": "에이징: 시간을 두고 향료들이 조화를 이루도록 하는 과정",
                    "filtering": "필터링: 불순물 제거 및 투명도 확보"
                }
            },
            "formulation": {
                "concentration": {
                    "parfum": "퍼퓸(20-30%): 가장 진한 농도, 8시간 이상 지속",
                    "edp": "오 드 퍼퓸(15-20%): 높은 농도, 4-6시간 지속",
                    "edt": "오 드 뚜왈렛(5-15%): 중간 농도, 2-4시간 지속",
                    "edc": "오 드 코롱(2-5%): 가벼운 농도, 1-2시간 지속",
                    "splash": "스플래시(1-3%): 매우 가벼운 농도"
                },
                "structure": {
                    "linear": "선형 구조: 시간에 따라 일정하게 향이 발산",
                    "non_linear": "비선형 구조: 시간에 따라 극적으로 변화하는 향",
                    "minimal": "미니멀: 적은 수의 노트로 구성된 단순한 구조",
                    "complex": "복합: 많은 노트가 어우러진 복잡한 구조"
                }
            }
        }
        return knowledge

    def _load_note_database(self) -> dict:
        """향료 노트 데이터베이스 로드"""
        notes = {
            "citrus": {
                "bergamot": {
                    "description": "얼그레이 티의 주 향료, 상쾌하고 약간 쓴맛",
                    "origin": "이탈리아 칼라브리아",
                    "pairs_well": ["lavender", "jasmine", "sandalwood"]
                },
                "lemon": {
                    "description": "밝고 상쾌한 시트러스, 청량감",
                    "origin": "이탈리아 시칠리아",
                    "pairs_well": ["mint", "basil", "white tea"]
                },
                "grapefruit": {
                    "description": "쓴맛과 단맛이 조화로운 과일향",
                    "origin": "미국 플로리다",
                    "pairs_well": ["rose", "black pepper", "vetiver"]
                }
            },
            "floral": {
                "rose": {
                    "description": "향수의 여왕, 로맨틱하고 우아한 향",
                    "origin": "불가리아, 터키",
                    "types": ["damascena", "centifolia"],
                    "pairs_well": ["oud", "patchouli", "vanilla"]
                },
                "jasmine": {
                    "description": "밤에 피는 꽃, 관능적이고 나르코틱한 향",
                    "origin": "인도, 이집트",
                    "types": ["grandiflorum", "sambac"],
                    "pairs_well": ["sandalwood", "bergamot", "ylang-ylang"]
                },
                "iris": {
                    "description": "파우더리하고 우아한 향, 럭셔리 향수의 핵심",
                    "origin": "이탈리아 피렌체",
                    "extraction": "뿌리를 3년간 건조",
                    "pairs_well": ["violet", "rose", "musk"]
                }
            },
            "woody": {
                "sandalwood": {
                    "description": "크리미하고 부드러운 우디향",
                    "origin": "인도 마이소르",
                    "sustainability": "멸종위기, 호주산 대체",
                    "pairs_well": ["rose", "jasmine", "vanilla"]
                },
                "cedar": {
                    "description": "드라이하고 연필 깎은 듯한 우디향",
                    "origin": "미국 버지니아",
                    "types": ["atlas", "virginia"],
                    "pairs_well": ["leather", "tobacco", "amber"]
                },
                "oud": {
                    "description": "액체 황금, 깊고 복잡한 동물성 우디향",
                    "origin": "캄보디아, 보르네오",
                    "price": "세계에서 가장 비싼 향료",
                    "pairs_well": ["rose", "saffron", "amber"]
                }
            },
            "oriental": {
                "vanilla": {
                    "description": "달콤하고 따뜻한 발사믹향",
                    "origin": "마다가스카르",
                    "extraction": "바닐라 포드 큐어링",
                    "pairs_well": ["tonka bean", "benzoin", "sandalwood"]
                },
                "amber": {
                    "description": "따뜻하고 레진향의 오리엔탈 베이스",
                    "composition": "labdanum + vanilla + benzoin",
                    "character": "파우더리, 스윗, 발사믹",
                    "pairs_well": ["patchouli", "musk", "incense"]
                },
                "incense": {
                    "description": "신성하고 명상적인 레진향",
                    "types": ["frankincense", "myrrh"],
                    "origin": "오만, 소말리아",
                    "pairs_well": ["rose", "oud", "labdanum"]
                }
            }
        }
        return notes

    def _load_accord_formulas(self) -> dict:
        """클래식 어코드 포뮬라 로드"""
        accords = {
            "chypre": {
                "description": "시프레 - 오크모스 베이스의 클래식 어코드",
                "formula": {
                    "bergamot": 30,
                    "oakmoss": 25,
                    "labdanum": 15,
                    "patchouli": 10,
                    "other": 20
                },
                "character": "모시, 어시, 엘레강트"
            },
            "fougere": {
                "description": "푸제르 - 라벤더와 쿠마린의 남성적 어코드",
                "formula": {
                    "lavender": 30,
                    "geranium": 20,
                    "coumarin": 20,
                    "oakmoss": 15,
                    "other": 15
                },
                "character": "프레시, 허벌, 마스큘린"
            },
            "oriental": {
                "description": "오리엔탈 - 따뜻하고 스파이시한 어코드",
                "formula": {
                    "vanilla": 25,
                    "amber": 20,
                    "spices": 20,
                    "resins": 15,
                    "other": 20
                },
                "character": "웜, 스파이시, 센슈얼"
            },
            "gourmand": {
                "description": "구르망 - 먹을 수 있는 달콤한 어코드",
                "formula": {
                    "vanilla": 30,
                    "caramel": 20,
                    "chocolate": 15,
                    "praline": 15,
                    "other": 20
                },
                "character": "스위트, 에디블, 컴포팅"
            }
        }
        return accords

    def _load_perfumer_styles(self) -> dict:
        """유명 조향사 스타일 로드"""
        styles = {
            "jean_claude_ellena": {
                "philosophy": "미니멀리즘, 투명성, 수채화적 표현",
                "signature": "가벼운 시트러스, 차, 투명한 플로럴",
                "famous_works": ["Terre d'Hermès", "Un Jardin series"],
                "technique": "적은 재료로 최대 효과"
            },
            "francis_kurkdjian": {
                "philosophy": "모던 엘레강스, 균형과 조화",
                "signature": "밝은 플로럴, 클린 머스크",
                "famous_works": ["Baccarat Rouge 540", "Aqua series"],
                "technique": "전통과 혁신의 조화"
            },
            "olivier_polge": {
                "philosophy": "전통 프렌치 스타일의 현대적 해석",
                "signature": "파우더리 아이리스, 빈티지 플로럴",
                "famous_works": ["Chanel No.5 L'Eau", "Gabrielle"],
                "technique": "클래식의 재해석"
            },
            "alberto_morillas": {
                "philosophy": "감성적이고 기억에 남는 향",
                "signature": "프루티 플로럴, 아쿠아틱",
                "famous_works": ["CK One", "Acqua di Gio"],
                "technique": "대중성과 예술성의 균형"
            }
        }
        return styles

    def query(self, query: KnowledgeQuery) -> KnowledgeResponse:
        """지식 쿼리 처리"""
        try:
            answer = ""
            sources = []
            related = []
            confidence = 0.0

            if query.category == "history":
                answer = self._query_history(query.query)
                sources = ["Perfume History Database"]
                confidence = 0.9

            elif query.category == "technique":
                answer = self._query_technique(query.query)
                sources = ["Technical Manual", "Industry Standards"]
                confidence = 0.95

            elif query.category == "note":
                answer = self._query_notes(query.query)
                sources = ["Note Database", "Raw Material Catalog"]
                confidence = 0.9

            elif query.category == "accord":
                answer = self._query_accords(query.query)
                sources = ["Classic Accord Formulas"]
                confidence = 0.85

            elif query.category == "perfumer":
                answer = self._query_perfumers(query.query)
                sources = ["Perfumer Profiles", "Industry Analysis"]
                confidence = 0.8

            else:
                answer = "카테고리를 찾을 수 없습니다."
                confidence = 0.0

            # 관련 주제 추출
            related = self._find_related_topics(query.category, query.query)

            return KnowledgeResponse(
                category=query.category,
                query=query.query,
                answer=answer,
                confidence=confidence,
                sources=sources,
                related_topics=related
            )

        except Exception as e:
            logger.error(f"Knowledge query failed: {e}")
            return KnowledgeResponse(
                category=query.category,
                query=query.query,
                answer="죄송합니다. 정보를 찾을 수 없습니다.",
                confidence=0.0,
                sources=[],
                related_topics=[]
            )

    def _query_history(self, query: str) -> str:
        """역사 관련 쿼리"""
        query_lower = query.lower()

        for period, info in self.knowledge_data["history"].items():
            for region, description in info.items():
                if region in query_lower or period in query_lower:
                    return description

        return "향수의 역사는 고대 이집트부터 시작되어 현재까지 이어지고 있습니다."

    def _query_technique(self, query: str) -> str:
        """기술 관련 쿼리"""
        query_lower = query.lower()

        for category, techniques in self.knowledge_data["techniques"].items():
            for technique, description in techniques.items():
                if technique in query_lower or category in query_lower:
                    return description

        return "향수 제조에는 추출, 블렌딩, 숙성 등 다양한 기술이 사용됩니다."

    def _query_notes(self, query: str) -> str:
        """노트 관련 쿼리"""
        query_lower = query.lower()

        for family, notes in self.note_database.items():
            for note_name, info in notes.items():
                if note_name in query_lower:
                    desc = info.get("description", "")
                    origin = info.get("origin", "")
                    pairs = info.get("pairs_well", [])

                    response = f"{note_name.title()}: {desc}"
                    if origin:
                        response += f"\n원산지: {origin}"
                    if pairs:
                        response += f"\n조화로운 조합: {', '.join(pairs)}"

                    return response

        return "해당 노트 정보를 찾을 수 없습니다."

    def _query_accords(self, query: str) -> str:
        """어코드 관련 쿼리"""
        query_lower = query.lower()

        for accord_name, info in self.accord_formulas.items():
            if accord_name in query_lower:
                desc = info.get("description", "")
                formula = info.get("formula", {})
                character = info.get("character", "")

                response = f"{desc}\n"
                response += f"특징: {character}\n"
                response += "주요 구성:\n"
                for ingredient, percentage in formula.items():
                    response += f"  - {ingredient}: {percentage}%\n"

                return response

        return "클래식 어코드 정보를 제공합니다."

    def _query_perfumers(self, query: str) -> str:
        """조향사 관련 쿼리"""
        query_lower = query.lower()

        for perfumer, info in self.perfumer_styles.items():
            perfumer_name = perfumer.replace("_", " ")
            if perfumer_name in query_lower:
                philosophy = info.get("philosophy", "")
                signature = info.get("signature", "")
                works = info.get("famous_works", [])

                response = f"{perfumer_name.title()}:\n"
                response += f"철학: {philosophy}\n"
                response += f"시그니처: {signature}\n"
                if works:
                    response += f"대표작: {', '.join(works)}"

                return response

        return "유명 조향사들의 스타일과 철학을 제공합니다."

    def _find_related_topics(self, category: str, query: str) -> List[str]:
        """관련 주제 찾기"""
        related = []

        if category == "note":
            # 같은 패밀리의 다른 노트들
            for family, notes in self.note_database.items():
                if any(note in query.lower() for note in notes.keys()):
                    related.extend([n for n in notes.keys() if n not in query.lower()][:3])
                    break

        elif category == "accord":
            # 다른 클래식 어코드들
            related = [a for a in self.accord_formulas.keys() if a not in query.lower()][:3]

        elif category == "perfumer":
            # 다른 조향사들
            related = [p.replace("_", " ").title()
                      for p in self.perfumer_styles.keys()
                      if p not in query.lower()][:3]

        return related

# 전역 지식 베이스 인스턴스
knowledge_base = None

def get_knowledge_base():
    """지식 베이스 인스턴스 가져오기"""
    global knowledge_base
    if knowledge_base is None:
        knowledge_base = PerfumeKnowledgeBase()
    return knowledge_base

async def query_knowledge_base(
    category: str,
    query: str,
    context: Optional[Dict[str, Any]] = None
) -> KnowledgeResponse:
    """
    # LLM TOOL DESCRIPTION (FOR ORCHESTRATOR)
    # Use this tool to access perfume domain knowledge including:
    # - History of perfumery
    # - Manufacturing techniques
    # - Note information and characteristics
    # - Classic accord formulas
    # - Famous perfumer styles

    Args:
        category: 지식 카테고리 (history, technique, note, accord, perfumer)
        query: 구체적인 질문
        context: 추가 컨텍스트 (선택사항)

    Returns:
        KnowledgeResponse: 지식 베이스 응답
    """
    kb = get_knowledge_base()
    knowledge_query = KnowledgeQuery(
        category=category,
        query=query,
        context=context
    )
    return kb.query(knowledge_query)