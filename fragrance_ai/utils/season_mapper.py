"""
계절 매핑 유틸리티
정확한 계절-향조 매칭을 위한 중앙 집중식 매핑
"""

from typing import List, Dict, Optional, Tuple
from enum import Enum


class Season(Enum):
    SPRING = "spring"
    SUMMER = "summer"
    AUTUMN = "autumn"
    FALL = "fall"  # Alias for autumn
    WINTER = "winter"
    ALL_SEASONS = "all_seasons"


class SeasonMapper:
    """계절과 향수 특성 매핑"""

    def __init__(self):
        # 계절별 대표 향조
        self.season_families = {
            Season.SPRING: ["floral", "green", "fresh", "light", "citrus"],
            Season.SUMMER: ["citrus", "aquatic", "fresh", "fruity", "marine"],
            Season.AUTUMN: ["woody", "oriental", "spicy", "warm", "amber"],
            Season.WINTER: ["oriental", "woody", "spicy", "vanilla", "musk", "amber", "leather"],
            Season.ALL_SEASONS: ["versatile", "balanced", "neutral"]
        }

        # 계절별 대표 노트
        self.season_notes = {
            Season.SPRING: {
                "top": ["bergamot", "lemon", "green leaves", "peach", "apple"],
                "heart": ["rose", "jasmine", "lily", "peony", "magnolia"],
                "base": ["white musk", "light woods", "green tea", "soft amber"]
            },
            Season.SUMMER: {
                "top": ["lemon", "lime", "grapefruit", "mint", "marine"],
                "heart": ["neroli", "orange blossom", "coconut", "watermelon"],
                "base": ["light musk", "driftwood", "sea salt", "white amber"]
            },
            Season.AUTUMN: {
                "top": ["bergamot", "orange", "cinnamon", "cardamom"],
                "heart": ["rose", "jasmine", "orchid", "plum", "fig"],
                "base": ["sandalwood", "patchouli", "amber", "tonka bean"]
            },
            Season.WINTER: {
                "top": ["bergamot", "black pepper", "pink pepper", "cinnamon"],
                "heart": ["rose", "jasmine", "tuberose", "orchid", "iris"],
                "base": ["vanilla", "amber", "musk", "oud", "sandalwood", "benzoin", "leather", "incense"]
            }
        }

        # 계절별 특성
        self.season_characteristics = {
            Season.SPRING: {
                "intensity": "light",
                "sillage": "moderate",
                "longevity": "4-6 hours",
                "mood": ["fresh", "romantic", "delicate", "optimistic"]
            },
            Season.SUMMER: {
                "intensity": "light",
                "sillage": "light to moderate",
                "longevity": "3-5 hours",
                "mood": ["energetic", "refreshing", "casual", "playful"]
            },
            Season.AUTUMN: {
                "intensity": "moderate",
                "sillage": "moderate to strong",
                "longevity": "6-8 hours",
                "mood": ["warm", "sophisticated", "mysterious", "cozy"]
            },
            Season.WINTER: {
                "intensity": "strong",
                "sillage": "strong",
                "longevity": "8-12 hours",
                "mood": ["warm", "sensual", "luxurious", "mysterious", "intimate"]
            }
        }

        # 계절 키워드 매핑 (사용자 입력 해석용)
        self.season_keywords = {
            Season.SPRING: ["봄", "spring", "벚꽃", "cherry blossom", "fresh start", "새로운", "시작"],
            Season.SUMMER: ["여름", "summer", "바다", "beach", "ocean", "휴가", "vacation", "시원한"],
            Season.AUTUMN: ["가을", "autumn", "fall", "단풍", "maple", "따뜻한", "warm", "코지"],
            Season.WINTER: ["겨울", "winter", "크리스마스", "christmas", "눈", "snow", "따뜻한", "포근한", "진한"]
        }

    def identify_season_from_text(self, text: str) -> Optional[Season]:
        """텍스트에서 계절 식별"""
        text_lower = text.lower()

        for season, keywords in self.season_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return season

        return None

    def get_recommended_season_for_notes(self, notes: Dict[str, List[str]]) -> List[Season]:
        """노트 구성에 따른 추천 계절"""
        recommended_seasons = []

        # 각 계절별로 매칭 점수 계산
        season_scores = {}

        for season in [Season.SPRING, Season.SUMMER, Season.AUTUMN, Season.WINTER]:
            score = 0
            season_note_set = self.season_notes[season]

            # 각 노트 레벨에서 매칭 확인
            for note_level, note_list in notes.items():
                if note_level in season_note_set:
                    for note in note_list:
                        note_lower = note.lower()
                        for season_note in season_note_set[note_level]:
                            if season_note in note_lower or note_lower in season_note:
                                score += 1

            season_scores[season] = score

        # 점수가 높은 계절들 선택
        max_score = max(season_scores.values()) if season_scores else 0
        if max_score > 0:
            for season, score in season_scores.items():
                if score >= max_score * 0.7:  # 최고 점수의 70% 이상인 계절들
                    recommended_seasons.append(season)

        return recommended_seasons if recommended_seasons else [Season.ALL_SEASONS]

    def get_season_appropriate_notes(self, season: Season, note_type: str = "all") -> Dict[str, List[str]]:
        """계절에 적합한 노트 반환"""
        if season not in self.season_notes:
            season = Season.ALL_SEASONS

        if season == Season.ALL_SEASONS:
            # 모든 계절에 어울리는 균형잡힌 노트
            return {
                "top": ["bergamot", "lemon", "lavender"],
                "heart": ["rose", "jasmine", "geranium"],
                "base": ["sandalwood", "musk", "amber"]
            }

        notes = self.season_notes[season]

        if note_type == "all":
            return notes
        elif note_type in notes:
            return {note_type: notes[note_type]}
        else:
            return {}

    def correct_season_mismatch(self, requested_season: str, generated_notes: Dict[str, List[str]]) -> Tuple[Dict[str, List[str]], str]:
        """
        요청된 계절과 생성된 노트의 불일치를 수정

        Args:
            requested_season: 사용자가 요청한 계절
            generated_notes: 생성된 노트 구성

        Returns:
            수정된 노트와 추천 계절 문자열
        """
        # 요청된 계절 식별
        season = self.identify_season_from_text(requested_season)

        if season == Season.WINTER:
            # 겨울 향수로 수정
            winter_notes = self.get_season_appropriate_notes(Season.WINTER)

            # 기존 노트와 겨울 노트를 적절히 혼합
            corrected_notes = {
                "top": winter_notes["top"][:3],  # 겨울 탑노트 사용
                "heart": winter_notes["heart"][:3],  # 겨울 하트노트 사용
                "base": winter_notes["base"][:4]  # 겨울 베이스노트 강화
            }

            recommended_season = "가을/겨울"

        elif season == Season.SUMMER:
            # 여름 향수로 수정
            summer_notes = self.get_season_appropriate_notes(Season.SUMMER)

            corrected_notes = {
                "top": summer_notes["top"][:3],
                "heart": summer_notes["heart"][:3],
                "base": summer_notes["base"][:3]
            }

            recommended_season = "봄/여름"

        elif season == Season.SPRING:
            spring_notes = self.get_season_appropriate_notes(Season.SPRING)

            corrected_notes = {
                "top": spring_notes["top"][:3],
                "heart": spring_notes["heart"][:3],
                "base": spring_notes["base"][:3]
            }

            recommended_season = "봄/초여름"

        elif season == Season.AUTUMN:
            autumn_notes = self.get_season_appropriate_notes(Season.AUTUMN)

            corrected_notes = {
                "top": autumn_notes["top"][:3],
                "heart": autumn_notes["heart"][:3],
                "base": autumn_notes["base"][:3]
            }

            recommended_season = "가을"

        else:
            # 계절 미지정시 원본 유지
            corrected_notes = generated_notes
            recommended_season = "사계절"

        return corrected_notes, recommended_season

    def get_season_intensity(self, season: Season) -> Dict[str, str]:
        """계절에 맞는 강도 특성 반환"""
        if season in self.season_characteristics:
            return {
                "intensity": self.season_characteristics[season]["intensity"],
                "sillage": self.season_characteristics[season]["sillage"],
                "longevity": self.season_characteristics[season]["longevity"]
            }

        return {
            "intensity": "moderate",
            "sillage": "moderate",
            "longevity": "6-8 hours"
        }


# 전역 인스턴스
season_mapper = SeasonMapper()


def get_season_mapper() -> SeasonMapper:
    """계절 매퍼 인스턴스 반환"""
    return season_mapper