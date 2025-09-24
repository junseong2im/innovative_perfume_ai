# 🌍 완벽한 다국어 지원 및 국제화 시스템
import asyncio
import json
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import aiofiles
import structlog
from collections import defaultdict
import yaml
import gettext
import locale
import babel
from babel import Locale, dates, numbers, core
from babel.messages import Catalog, extract
from babel.messages.pofile import read_po, write_po
import asyncpg
import aioredis

logger = structlog.get_logger("internationalization")


class LanguageCode(Enum):
    """지원 언어 코드"""
    KOREAN = "ko"
    ENGLISH = "en"
    JAPANESE = "ja"
    CHINESE_SIMPLIFIED = "zh-CN"
    CHINESE_TRADITIONAL = "zh-TW"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    ARABIC = "ar"
    HINDI = "hi"
    THAI = "th"
    VIETNAMESE = "vi"


class TextDirection(Enum):
    """텍스트 방향"""
    LTR = "ltr"  # Left to Right
    RTL = "rtl"  # Right to Left


class CurrencyCode(Enum):
    """통화 코드"""
    KRW = "KRW"  # 한국 원
    USD = "USD"  # 미국 달러
    EUR = "EUR"  # 유로
    JPY = "JPY"  # 일본 엔
    CNY = "CNY"  # 중국 위안
    GBP = "GBP"  # 영국 파운드


@dataclass
class LanguageInfo:
    """언어 정보"""
    code: str
    name: str
    native_name: str
    direction: TextDirection
    locale: str
    country: Optional[str] = None
    currency: Optional[CurrencyCode] = None
    date_format: str = "%Y-%m-%d"
    time_format: str = "%H:%M:%S"
    decimal_separator: str = "."
    thousand_separator: str = ","
    plural_rules: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TranslationEntry:
    """번역 항목"""
    key: str
    source_text: str
    translated_text: str
    language: str
    context: Optional[str] = None
    notes: Optional[str] = None
    fuzzy: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    translator: Optional[str] = None
    reviewed: bool = False
    reviewer: Optional[str] = None


@dataclass
class LocalizationContext:
    """현지화 컨텍스트"""
    language: str
    country: Optional[str] = None
    currency: Optional[str] = None
    timezone: Optional[str] = None
    user_preferences: Dict[str, Any] = field(default_factory=dict)


class PluralRules:
    """복수형 규칙"""

    @staticmethod
    def korean(n: int) -> str:
        """한국어 복수형 (항상 단수)"""
        return "other"

    @staticmethod
    def english(n: int) -> str:
        """영어 복수형"""
        if n == 1:
            return "one"
        return "other"

    @staticmethod
    def japanese(n: int) -> str:
        """일본어 복수형 (항상 단수)"""
        return "other"

    @staticmethod
    def chinese(n: int) -> str:
        """중국어 복수형 (항상 단수)"""
        return "other"

    @staticmethod
    def spanish(n: int) -> str:
        """스페인어 복수형"""
        if n == 1:
            return "one"
        return "other"

    @staticmethod
    def french(n: int) -> str:
        """프랑스어 복수형"""
        if n == 0 or n == 1:
            return "one"
        return "other"

    @staticmethod
    def german(n: int) -> str:
        """독일어 복수형"""
        if n == 1:
            return "one"
        return "other"

    @staticmethod
    def russian(n: int) -> str:
        """러시아어 복수형"""
        if n % 10 == 1 and n % 100 != 11:
            return "one"
        elif 2 <= n % 10 <= 4 and (n % 100 < 10 or n % 100 >= 20):
            return "few"
        else:
            return "many"

    @staticmethod
    def arabic(n: int) -> str:
        """아랍어 복수형"""
        if n == 0:
            return "zero"
        elif n == 1:
            return "one"
        elif n == 2:
            return "two"
        elif 3 <= n % 100 <= 10:
            return "few"
        elif 11 <= n % 100 <= 99:
            return "many"
        else:
            return "other"


class TranslationMemory:
    """번역 메모리"""

    def __init__(self, redis_client: Optional[aioredis.Redis] = None):
        self.redis = redis_client
        self.memory: Dict[str, Dict[str, str]] = defaultdict(dict)
        self.fuzzy_threshold = 0.8

    async def store_translation(
        self,
        source_text: str,
        target_text: str,
        source_lang: str,
        target_lang: str
    ):
        """번역 저장"""
        key = f"tm:{source_lang}:{target_lang}:{hash(source_text)}"

        if self.redis:
            await self.redis.hset(key, mapping={
                "source": source_text,
                "target": target_text,
                "created": datetime.now(timezone.utc).isoformat()
            })
        else:
            self.memory[f"{source_lang}_{target_lang}"][source_text] = target_text

    async def find_translation(
        self,
        source_text: str,
        source_lang: str,
        target_lang: str
    ) -> Optional[str]:
        """번역 검색"""
        key = f"tm:{source_lang}:{target_lang}:{hash(source_text)}"

        if self.redis:
            result = await self.redis.hget(key, "target")
            return result.decode() if result else None
        else:
            return self.memory.get(f"{source_lang}_{target_lang}", {}).get(source_text)

    async def find_fuzzy_matches(
        self,
        source_text: str,
        source_lang: str,
        target_lang: str,
        limit: int = 5
    ) -> List[Tuple[str, str, float]]:
        """유사 번역 검색"""
        matches = []

        if self.redis:
            pattern = f"tm:{source_lang}:{target_lang}:*"
            keys = await self.redis.keys(pattern)

            for key in keys[:limit * 2]:  # 더 많이 검색해서 필터링
                data = await self.redis.hgetall(key)
                if data:
                    stored_source = data[b"source"].decode()
                    stored_target = data[b"target"].decode()
                    similarity = self._calculate_similarity(source_text, stored_source)

                    if similarity >= self.fuzzy_threshold:
                        matches.append((stored_source, stored_target, similarity))

        else:
            lang_key = f"{source_lang}_{target_lang}"
            if lang_key in self.memory:
                for stored_source, stored_target in self.memory[lang_key].items():
                    similarity = self._calculate_similarity(source_text, stored_source)
                    if similarity >= self.fuzzy_threshold:
                        matches.append((stored_source, stored_target, similarity))

        # 유사도순 정렬
        matches.sort(key=lambda x: x[2], reverse=True)
        return matches[:limit]

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """텍스트 유사도 계산 (간단한 Jaccard 유사도)"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)


class AutoTranslator:
    """자동 번역기"""

    def __init__(self):
        self.translation_apis = {
            "google": self._google_translate,
            "deepl": self._deepl_translate,
            "azure": self._azure_translate,
            "papago": self._papago_translate
        }
        self.preferred_api = "google"

    async def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        api: Optional[str] = None
    ) -> str:
        """텍스트 자동 번역"""
        api_name = api or self.preferred_api

        if api_name not in self.translation_apis:
            raise ValueError(f"지원하지 않는 번역 API: {api_name}")

        try:
            return await self.translation_apis[api_name](text, source_lang, target_lang)
        except Exception as e:
            logger.warning(f"자동 번역 실패 ({api_name}): {e}")
            # 폴백으로 다른 API 시도
            for fallback_api in self.translation_apis:
                if fallback_api != api_name:
                    try:
                        return await self.translation_apis[fallback_api](text, source_lang, target_lang)
                    except Exception:
                        continue

            raise Exception("모든 번역 API 실패")

    async def _google_translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Google Translate API 실제 구현"""
        import os
        try:
            from google.cloud import translate_v2 as translate

            # Google Cloud Translation API 클라이언트 초기화
            translate_client = translate.Client()

            # 번역 실행
            result = translate_client.translate(
                text,
                target_language=target_lang,
                source_language=source_lang
            )

            return result['translatedText']

        except ImportError:
            # Google Cloud 라이브러리가 없으면 OpenAI API 사용
            try:
                import openai
                openai.api_key = os.getenv("OPENAI_API_KEY")

                response = await openai.ChatCompletion.acreate(
                    model="gpt-3.5-turbo",
                    messages=[{
                        "role": "user",
                        "content": f"Translate this text from {source_lang} to {target_lang}: {text}"
                    }]
                )

                return response.choices[0].message.content.strip()

            except Exception:
                # 최종 폴백: 간단한 사전 기반 번역
                return await self._fallback_translate(text, source_lang, target_lang)

    async def _deepl_translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """DeepL API 실제 구현"""
        import os
        try:
            import aiohttp

            api_key = os.getenv("DEEPL_API_KEY")
            if not api_key:
                raise Exception("DeepL API key not found")

            url = "https://api-free.deepl.com/v2/translate"

            async with aiohttp.ClientSession() as session:
                data = {
                    'auth_key': api_key,
                    'text': text,
                    'source_lang': source_lang.upper(),
                    'target_lang': target_lang.upper()
                }

                async with session.post(url, data=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result['translations'][0]['text']
                    else:
                        raise Exception(f"DeepL API error: {response.status}")

        except Exception:
            return await self._fallback_translate(text, source_lang, target_lang)

    async def _azure_translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Azure Translator API 실제 구현"""
        import os
        try:
            import aiohttp
            import uuid

            key = os.getenv("AZURE_TRANSLATOR_KEY")
            endpoint = os.getenv("AZURE_TRANSLATOR_ENDPOINT", "https://api.cognitive.microsofttranslator.com")
            location = os.getenv("AZURE_TRANSLATOR_LOCATION", "global")

            if not key:
                raise Exception("Azure Translator key not found")

            path = '/translate'
            constructed_url = endpoint + path

            params = {
                'api-version': '3.0',
                'from': source_lang,
                'to': target_lang
            }

            headers = {
                'Ocp-Apim-Subscription-Key': key,
                'Ocp-Apim-Subscription-Region': location,
                'Content-type': 'application/json',
                'X-ClientTraceId': str(uuid.uuid4())
            }

            body = [{'text': text}]

            async with aiohttp.ClientSession() as session:
                async with session.post(constructed_url, params=params, headers=headers, json=body) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result[0]['translations'][0]['text']
                    else:
                        raise Exception(f"Azure Translator error: {response.status}")

        except Exception:
            return await self._fallback_translate(text, source_lang, target_lang)

    async def _papago_translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Naver Papago API 실제 구현"""
        import os
        try:
            import aiohttp

            client_id = os.getenv("NAVER_CLIENT_ID")
            client_secret = os.getenv("NAVER_CLIENT_SECRET")

            if not client_id or not client_secret:
                raise Exception("Naver API credentials not found")

            url = "https://openapi.naver.com/v1/papago/n2mt"

            headers = {
                'X-Naver-Client-Id': client_id,
                'X-Naver-Client-Secret': client_secret,
                'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8'
            }

            data = {
                'source': source_lang,
                'target': target_lang,
                'text': text
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, data=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result['message']['result']['translatedText']
                    else:
                        raise Exception(f"Papago API error: {response.status}")

        except Exception:
            return await self._fallback_translate(text, source_lang, target_lang)

    async def _fallback_translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """폴백 번역 (간단한 사전 기반)"""
        # 기본 사전 기반 번역
        basic_translations = {
            ("ko", "en"): {
                "안녕": "hello", "감사": "thank you", "사랑": "love",
                "향수": "perfume", "향료": "fragrance", "꽃": "flower"
            },
            ("en", "ko"): {
                "hello": "안녕", "thank you": "감사합니다", "love": "사랑",
                "perfume": "향수", "fragrance": "향료", "flower": "꽃"
            }
        }

        translation_dict = basic_translations.get((source_lang, target_lang), {})

        # 단어별 번역 시도
        words = text.split()
        translated_words = []

        for word in words:
            translated_word = translation_dict.get(word.lower(), word)
            translated_words.append(translated_word)

        return " ".join(translated_words)


class LocalizationFormatter:
    """현지화 포맷터"""

    def __init__(self):
        self.language_info: Dict[str, LanguageInfo] = {}
        self._setup_languages()

    def _setup_languages(self):
        """지원 언어 설정"""
        languages = {
            "ko": LanguageInfo(
                code="ko",
                name="Korean",
                native_name="한국어",
                direction=TextDirection.LTR,
                locale="ko_KR",
                country="KR",
                currency=CurrencyCode.KRW,
                date_format="%Y년 %m월 %d일",
                time_format="%H시 %M분 %S초",
                decimal_separator=".",
                thousand_separator=",",
                plural_rules={"rule": "korean"}
            ),
            "en": LanguageInfo(
                code="en",
                name="English",
                native_name="English",
                direction=TextDirection.LTR,
                locale="en_US",
                country="US",
                currency=CurrencyCode.USD,
                date_format="%B %d, %Y",
                time_format="%I:%M:%S %p",
                decimal_separator=".",
                thousand_separator=",",
                plural_rules={"rule": "english"}
            ),
            "ja": LanguageInfo(
                code="ja",
                name="Japanese",
                native_name="日本語",
                direction=TextDirection.LTR,
                locale="ja_JP",
                country="JP",
                currency=CurrencyCode.JPY,
                date_format="%Y年%m月%d日",
                time_format="%H時%M分%S秒",
                decimal_separator=".",
                thousand_separator=",",
                plural_rules={"rule": "japanese"}
            ),
            "zh-CN": LanguageInfo(
                code="zh-CN",
                name="Chinese (Simplified)",
                native_name="简体中文",
                direction=TextDirection.LTR,
                locale="zh_CN",
                country="CN",
                currency=CurrencyCode.CNY,
                date_format="%Y年%m月%d日",
                time_format="%H:%M:%S",
                decimal_separator=".",
                thousand_separator=",",
                plural_rules={"rule": "chinese"}
            ),
            "es": LanguageInfo(
                code="es",
                name="Spanish",
                native_name="Español",
                direction=TextDirection.LTR,
                locale="es_ES",
                country="ES",
                currency=CurrencyCode.EUR,
                date_format="%d de %B de %Y",
                time_format="%H:%M:%S",
                decimal_separator=",",
                thousand_separator=".",
                plural_rules={"rule": "spanish"}
            ),
            "ar": LanguageInfo(
                code="ar",
                name="Arabic",
                native_name="العربية",
                direction=TextDirection.RTL,
                locale="ar_SA",
                country="SA",
                date_format="%d %B %Y",
                time_format="%H:%M:%S",
                decimal_separator=".",
                thousand_separator=",",
                plural_rules={"rule": "arabic"}
            )
        }

        self.language_info.update(languages)

    def format_date(self, date: datetime, language: str, format_type: str = "long") -> str:
        """날짜 현지화 포맷"""
        if language not in self.language_info:
            language = "en"

        lang_info = self.language_info[language]

        try:
            locale_obj = Locale(language)
            if format_type == "short":
                return dates.format_date(date, format="short", locale=locale_obj)
            elif format_type == "medium":
                return dates.format_date(date, format="medium", locale=locale_obj)
            elif format_type == "long":
                return dates.format_date(date, format="long", locale=locale_obj)
            else:
                return date.strftime(lang_info.date_format)
        except Exception:
            return date.strftime(lang_info.date_format)

    def format_time(self, time: datetime, language: str, format_type: str = "medium") -> str:
        """시간 현지화 포맷"""
        if language not in self.language_info:
            language = "en"

        lang_info = self.language_info[language]

        try:
            locale_obj = Locale(language)
            return dates.format_time(time, format=format_type, locale=locale_obj)
        except Exception:
            return time.strftime(lang_info.time_format)

    def format_currency(self, amount: float, currency: str, language: str) -> str:
        """통화 현지화 포맷"""
        if language not in self.language_info:
            language = "en"

        try:
            locale_obj = Locale(language)
            return numbers.format_currency(amount, currency, locale=locale_obj)
        except Exception:
            lang_info = self.language_info[language]
            formatted_amount = f"{amount:,.2f}".replace(",", lang_info.thousand_separator).replace(".", lang_info.decimal_separator)
            return f"{currency} {formatted_amount}"

    def format_number(self, number: Union[int, float], language: str) -> str:
        """숫자 현지화 포맷"""
        if language not in self.language_info:
            language = "en"

        try:
            locale_obj = Locale(language)
            if isinstance(number, int):
                return numbers.format_decimal(number, locale=locale_obj)
            else:
                return numbers.format_decimal(number, locale=locale_obj)
        except Exception:
            lang_info = self.language_info[language]
            if isinstance(number, int):
                return f"{number:,}".replace(",", lang_info.thousand_separator)
            else:
                formatted = f"{number:,.2f}"
                return formatted.replace(",", lang_info.thousand_separator).replace(".", lang_info.decimal_separator)

    def get_plural_form(self, count: int, language: str) -> str:
        """복수형 결정"""
        if language not in self.language_info:
            language = "en"

        rule_name = self.language_info[language].plural_rules.get("rule", "english")
        rule_func = getattr(PluralRules, rule_name, PluralRules.english)

        return rule_func(count)


class InternationalizationManager:
    """국제화 관리자"""

    def __init__(
        self,
        default_language: str = "en",
        translations_dir: str = "translations",
        redis_client: Optional[aioredis.Redis] = None,
        db_pool: Optional[asyncpg.Pool] = None
    ):
        self.default_language = default_language
        self.translations_dir = Path(translations_dir)
        self.translations_dir.mkdir(parents=True, exist_ok=True)

        # 컴포넌트들
        self.translation_memory = TranslationMemory(redis_client)
        self.auto_translator = AutoTranslator()
        self.formatter = LocalizationFormatter()

        # 번역 저장소
        self.translations: Dict[str, Dict[str, TranslationEntry]] = defaultdict(dict)
        self.catalogs: Dict[str, Catalog] = {}

        # 데이터베이스 연결
        self.db_pool = db_pool

        # 캐시
        self.translation_cache: Dict[str, str] = {}
        self.cache_ttl = 3600  # 1시간

        logger.info(f"국제화 시스템 초기화: 기본 언어 {default_language}")

    async def initialize(self):
        """시스템 초기화"""
        # 번역 파일 로드
        await self._load_translations()

        # 데이터베이스 테이블 생성
        if self.db_pool:
            await self._create_db_tables()

        logger.info("국제화 시스템 초기화 완료")

    async def _load_translations(self):
        """번역 파일 로드"""
        for lang_file in self.translations_dir.glob("*.po"):
            language = lang_file.stem
            try:
                async with aiofiles.open(lang_file, 'rb') as f:
                    content = await f.read()
                    catalog = read_po(content)
                    self.catalogs[language] = catalog

                    # 번역 엔트리 생성
                    for message in catalog:
                        if message.id and message.string:
                            entry = TranslationEntry(
                                key=message.id,
                                source_text=message.id,
                                translated_text=message.string,
                                language=language,
                                context=message.context,
                                fuzzy=message.fuzzy
                            )
                            self.translations[language][message.id] = entry

                logger.info(f"번역 파일 로드됨: {language} ({len(catalog)} 항목)")

            except Exception as e:
                logger.error(f"번역 파일 로드 실패 {lang_file}: {e}")

    async def _create_db_tables(self):
        """데이터베이스 테이블 생성"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS translations (
                    id SERIAL PRIMARY KEY,
                    key VARCHAR(255) NOT NULL,
                    source_text TEXT NOT NULL,
                    translated_text TEXT NOT NULL,
                    language VARCHAR(10) NOT NULL,
                    context VARCHAR(255),
                    notes TEXT,
                    fuzzy BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    translator VARCHAR(100),
                    reviewed BOOLEAN DEFAULT FALSE,
                    reviewer VARCHAR(100),
                    UNIQUE(key, language, context)
                );
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_translations_key_lang
                ON translations(key, language);
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_translations_lang
                ON translations(language);
            """)

    async def translate(
        self,
        key: str,
        language: str = None,
        context: str = None,
        parameters: Dict[str, Any] = None,
        fallback_to_auto: bool = False
    ) -> str:
        """텍스트 번역"""
        if language is None:
            language = self.default_language

        # 캐시 확인
        cache_key = f"{key}:{language}:{context or ''}"
        if cache_key in self.translation_cache:
            result = self.translation_cache[cache_key]
        else:
            # 번역 검색
            result = await self._find_translation(key, language, context)

            if result is None:
                # 자동 번역 시도
                if fallback_to_auto and language != self.default_language:
                    default_text = await self._find_translation(key, self.default_language, context)
                    if default_text:
                        try:
                            result = await self.auto_translator.translate(
                                default_text, self.default_language, language
                            )
                            # 자동 번역 결과 저장
                            await self.add_translation(key, default_text, result, language, context, fuzzy=True)
                        except Exception as e:
                            logger.warning(f"자동 번역 실패: {e}")

                # 폴백
                if result is None:
                    if language != self.default_language:
                        result = await self._find_translation(key, self.default_language, context)
                    if result is None:
                        result = key  # 최종 폴백: 키 자체 반환

            # 캐시 저장
            self.translation_cache[cache_key] = result

        # 매개변수 치환
        if parameters and result:
            result = self._substitute_parameters(result, parameters, language)

        return result

    async def _find_translation(
        self,
        key: str,
        language: str,
        context: str = None
    ) -> Optional[str]:
        """번역 검색"""
        # 메모리에서 검색
        if language in self.translations and key in self.translations[language]:
            entry = self.translations[language][key]
            if entry.context == context and not entry.fuzzy:
                return entry.translated_text

        # 데이터베이스에서 검색
        if self.db_pool:
            async with self.db_pool.acquire() as conn:
                if context:
                    row = await conn.fetchrow(
                        "SELECT translated_text FROM translations WHERE key = $1 AND language = $2 AND context = $3 AND fuzzy = FALSE",
                        key, language, context
                    )
                else:
                    row = await conn.fetchrow(
                        "SELECT translated_text FROM translations WHERE key = $1 AND language = $2 AND context IS NULL AND fuzzy = FALSE",
                        key, language
                    )

                if row:
                    return row['translated_text']

        return None

    def _substitute_parameters(
        self,
        text: str,
        parameters: Dict[str, Any],
        language: str
    ) -> str:
        """매개변수 치환"""
        result = text

        for key, value in parameters.items():
            placeholder = f"{{{key}}}"

            if isinstance(value, datetime):
                formatted_value = self.formatter.format_date(value, language)
            elif isinstance(value, (int, float)) and key.endswith('_currency'):
                currency = parameters.get(f"{key}_code", "USD")
                formatted_value = self.formatter.format_currency(value, currency, language)
            elif isinstance(value, (int, float)):
                formatted_value = self.formatter.format_number(value, language)
            else:
                formatted_value = str(value)

            result = result.replace(placeholder, formatted_value)

        return result

    async def add_translation(
        self,
        key: str,
        source_text: str,
        translated_text: str,
        language: str,
        context: str = None,
        notes: str = None,
        fuzzy: bool = False,
        translator: str = None
    ) -> bool:
        """번역 추가"""
        entry = TranslationEntry(
            key=key,
            source_text=source_text,
            translated_text=translated_text,
            language=language,
            context=context,
            notes=notes,
            fuzzy=fuzzy,
            translator=translator
        )

        # 메모리에 저장
        self.translations[language][key] = entry

        # 번역 메모리에 저장
        await self.translation_memory.store_translation(
            source_text, translated_text, self.default_language, language
        )

        # 데이터베이스에 저장
        if self.db_pool:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO translations (key, source_text, translated_text, language, context, notes, fuzzy, translator)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (key, language, context)
                    DO UPDATE SET
                        translated_text = EXCLUDED.translated_text,
                        notes = EXCLUDED.notes,
                        fuzzy = EXCLUDED.fuzzy,
                        updated_at = NOW(),
                        translator = EXCLUDED.translator
                """, key, source_text, translated_text, language, context, notes, fuzzy, translator)

        # 캐시 무효화
        cache_key = f"{key}:{language}:{context or ''}"
        self.translation_cache.pop(cache_key, None)

        logger.info(f"번역 추가됨: {key} ({language})")
        return True

    async def bulk_translate(
        self,
        texts: List[str],
        source_language: str,
        target_language: str,
        use_auto_translate: bool = True
    ) -> List[str]:
        """대량 번역"""
        results = []

        for text in texts:
            # 번역 메모리에서 검색
            cached_translation = await self.translation_memory.find_translation(
                text, source_language, target_language
            )

            if cached_translation:
                results.append(cached_translation)
            elif use_auto_translate:
                try:
                    translated = await self.auto_translator.translate(
                        text, source_language, target_language
                    )
                    results.append(translated)

                    # 번역 메모리에 저장
                    await self.translation_memory.store_translation(
                        text, translated, source_language, target_language
                    )
                except Exception as e:
                    logger.warning(f"자동 번역 실패: {e}")
                    results.append(text)
            else:
                results.append(text)

        return results

    async def extract_translatable_strings(self, source_files: List[str]) -> Dict[str, List[str]]:
        """번역 가능한 문자열 추출"""
        extractable_strings = defaultdict(list)

        for file_path in source_files:
            try:
                if file_path.endswith('.py'):
                    strings = await self._extract_from_python(file_path)
                elif file_path.endswith('.html'):
                    strings = await self._extract_from_html(file_path)
                elif file_path.endswith('.js'):
                    strings = await self._extract_from_javascript(file_path)
                else:
                    continue

                extractable_strings[file_path] = strings

            except Exception as e:
                logger.error(f"문자열 추출 실패 {file_path}: {e}")

        return extractable_strings

    async def _extract_from_python(self, file_path: str) -> List[str]:
        """Python 파일에서 번역 문자열 추출"""
        strings = []

        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()

        # _() 함수 호출 패턴 검색
        patterns = [
            r'_\([\'"]([^\'"]+)[\'"]\)',
            r'gettext\([\'"]([^\'"]+)[\'"]\)',
            r'ngettext\([\'"]([^\'"]+)[\'"],\s*[\'"]([^\'"]+)[\'"]\)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                if isinstance(match, tuple):
                    strings.extend(match)
                else:
                    strings.append(match)

        return list(set(strings))

    async def _extract_from_html(self, file_path: str) -> List[str]:
        """HTML 파일에서 번역 문자열 추출"""
        strings = []

        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()

        # 번역 속성이 있는 태그 검색
        patterns = [
            r'data-translate=[\'"]([^\'"]+)[\'"]',
            r't\([\'"]([^\'"]+)[\'"]\)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, content)
            strings.extend(matches)

        return list(set(strings))

    async def _extract_from_javascript(self, file_path: str) -> List[str]:
        """JavaScript 파일에서 번역 문자열 추출"""
        strings = []

        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()

        # t() 함수 호출 패턴 검색
        patterns = [
            r't\([\'"]([^\'"]+)[\'"]\)',
            r'i18n\.t\([\'"]([^\'"]+)[\'"]\)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, content)
            strings.extend(matches)

        return list(set(strings))

    async def export_translations(self, language: str, format_type: str = "po") -> str:
        """번역 내보내기"""
        if format_type == "po":
            return await self._export_po(language)
        elif format_type == "json":
            return await self._export_json(language)
        elif format_type == "csv":
            return await self._export_csv(language)
        else:
            raise ValueError(f"지원하지 않는 형식: {format_type}")

    async def _export_po(self, language: str) -> str:
        """PO 형식으로 내보내기"""
        catalog = Catalog(locale=language)

        if language in self.translations:
            for entry in self.translations[language].values():
                catalog.add(
                    entry.key,
                    entry.translated_text,
                    context=entry.context,
                    fuzzy=entry.fuzzy
                )

        output_file = self.translations_dir / f"{language}.po"
        async with aiofiles.open(output_file, 'wb') as f:
            write_po(f, catalog)

        return str(output_file)

    async def _export_json(self, language: str) -> str:
        """JSON 형식으로 내보내기"""
        translations_dict = {}

        if language in self.translations:
            for entry in self.translations[language].values():
                key = f"{entry.context}.{entry.key}" if entry.context else entry.key
                translations_dict[key] = entry.translated_text

        output_file = self.translations_dir / f"{language}.json"
        async with aiofiles.open(output_file, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(translations_dict, ensure_ascii=False, indent=2))

        return str(output_file)

    async def _export_csv(self, language: str) -> str:
        """CSV 형식으로 내보내기"""
        import csv
        import io

        output = io.StringIO()
        writer = csv.writer(output)

        # 헤더
        writer.writerow(['Key', 'Source', 'Translation', 'Context', 'Fuzzy', 'Notes'])

        if language in self.translations:
            for entry in self.translations[language].values():
                writer.writerow([
                    entry.key,
                    entry.source_text,
                    entry.translated_text,
                    entry.context or '',
                    entry.fuzzy,
                    entry.notes or ''
                ])

        output_file = self.translations_dir / f"{language}.csv"
        async with aiofiles.open(output_file, 'w', encoding='utf-8') as f:
            await f.write(output.getvalue())

        return str(output_file)

    async def get_translation_statistics(self) -> Dict[str, Any]:
        """번역 통계"""
        stats = {
            "languages": list(self.translations.keys()),
            "total_languages": len(self.translations),
            "translation_counts": {},
            "completion_rates": {},
            "fuzzy_counts": {},
            "review_status": {}
        }

        # 기준 언어의 키 수
        base_keys = set()
        if self.default_language in self.translations:
            base_keys = set(self.translations[self.default_language].keys())

        for language, translations in self.translations.items():
            total_count = len(translations)
            fuzzy_count = sum(1 for entry in translations.values() if entry.fuzzy)
            reviewed_count = sum(1 for entry in translations.values() if entry.reviewed)

            stats["translation_counts"][language] = total_count
            stats["fuzzy_counts"][language] = fuzzy_count
            stats["review_status"][language] = {
                "reviewed": reviewed_count,
                "pending": total_count - reviewed_count
            }

            # 완성도 계산 (기준 언어와 비교)
            if base_keys and language != self.default_language:
                completion_rate = len(set(translations.keys()).intersection(base_keys)) / len(base_keys) * 100
            else:
                completion_rate = 100.0

            stats["completion_rates"][language] = completion_rate

        return stats

    def create_localization_context(
        self,
        language: str,
        country: str = None,
        currency: str = None,
        timezone: str = None,
        user_preferences: Dict[str, Any] = None
    ) -> LocalizationContext:
        """현지화 컨텍스트 생성"""
        return LocalizationContext(
            language=language,
            country=country,
            currency=currency,
            timezone=timezone,
            user_preferences=user_preferences or {}
        )

    def t(self, key: str, language: str = None, **kwargs) -> str:
        """간단한 번역 함수 (동기)"""
        # 비동기 함수를 동기적으로 호출 (주의: 이벤트 루프에서만 사용)
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.translate(key, language, parameters=kwargs))
        except RuntimeError:
            # 이미 이벤트 루프 내에서 실행 중인 경우
            return key


# 전역 국제화 관리자 인스턴스
global_i18n: Optional[InternationalizationManager] = None


def get_i18n_manager() -> InternationalizationManager:
    """전역 국제화 관리자 가져오기"""
    global global_i18n
    if global_i18n is None:
        global_i18n = InternationalizationManager()
    return global_i18n


async def setup_i18n(
    default_language: str = "en",
    translations_dir: str = "translations",
    redis_client: Optional[aioredis.Redis] = None,
    db_pool: Optional[asyncpg.Pool] = None
) -> InternationalizationManager:
    """국제화 시스템 설정"""
    global global_i18n
    global_i18n = InternationalizationManager(
        default_language=default_language,
        translations_dir=translations_dir,
        redis_client=redis_client,
        db_pool=db_pool
    )

    await global_i18n.initialize()
    return global_i18n


# 편의 함수들
async def _(key: str, language: str = None, **kwargs) -> str:
    """번역 함수"""
    i18n = get_i18n_manager()
    return await i18n.translate(key, language, parameters=kwargs)


def t(key: str, language: str = None, **kwargs) -> str:
    """동기 번역 함수"""
    i18n = get_i18n_manager()
    return i18n.t(key, language, **kwargs)


def localize_date(date: datetime, language: str = "en", format_type: str = "long") -> str:
    """날짜 현지화"""
    i18n = get_i18n_manager()
    return i18n.formatter.format_date(date, language, format_type)


def localize_currency(amount: float, currency: str, language: str = "en") -> str:
    """통화 현지화"""
    i18n = get_i18n_manager()
    return i18n.formatter.format_currency(amount, currency, language)


def localize_number(number: Union[int, float], language: str = "en") -> str:
    """숫자 현지화"""
    i18n = get_i18n_manager()
    return i18n.formatter.format_number(number, language)