"""
종합적인 데이터 파이프라인 및 검증 시스템
ETL, 데이터 품질 검증, 자동 정제, 모니터링
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timezone, timedelta
import json
import hashlib
import re
from pathlib import Path
import aiofiles
import aiohttp
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
try:
    import great_expectations as ge
    from great_expectations.core import ExpectationSuite, ExpectationContext
    from great_expectations.validator.validator import Validator
except ImportError:
    ge = None
try:
    from scipy import stats
except ImportError:
    stats = None
import nltk
from langdetect import detect
from transformers import pipeline
import logging
import time

# Optional imports - use fallbacks if not available
try:
    from ..core.config import settings
except ImportError:
    class Settings:
        database_url = "postgresql+asyncpg://localhost/fragrance_db"
    settings = Settings()

try:
    from ..core.advanced_logging import get_logger, LogContext
    logger = get_logger(__name__, LogContext.SYSTEM)
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

try:
    from ..core.exceptions import ValidationError, DataPipelineError
except ImportError:
    class ValidationError(Exception):
        pass
    class DataPipelineError(Exception):
        pass

try:
    from ..database.connection import get_async_session
except ImportError:
    def get_async_session():
        raise NotImplementedError("Database connection not configured")


class DataQualityLevel(str, Enum):
    """데이터 품질 레벨"""
    EXCELLENT = "excellent"    # 90-100%
    GOOD = "good"             # 70-89%
    FAIR = "fair"             # 50-69%
    POOR = "poor"             # 30-49%
    CRITICAL = "critical"     # 0-29%


class DataSourceType(str, Enum):
    """데이터 소스 타입"""
    CSV = "csv"
    JSON = "json"
    JSONL = "jsonl"
    DATABASE = "database"
    API = "api"
    EXCEL = "excel"
    PARQUET = "parquet"


class ValidationSeverity(str, Enum):
    """검증 심각도"""
    ERROR = "error"       # 처리 중단
    WARNING = "warning"   # 경고 후 계속
    INFO = "info"        # 정보성


@dataclass
class DataQualityMetrics:
    """데이터 품질 메트릭"""
    completeness: float         # 완전성 (비어있지 않은 값 비율)
    validity: float            # 유효성 (형식 규칙 준수 비율)
    consistency: float         # 일관성 (중복, 충돌 없는 비율)
    accuracy: float           # 정확성 (실제 값과의 일치 비율)
    uniqueness: float         # 고유성 (중복 없는 비율)
    timeliness: float         # 시의성 (최신성)
    overall_score: float      # 전체 점수
    quality_level: DataQualityLevel
    issues_count: Dict[str, int] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class ValidationRule:
    """검증 규칙"""
    name: str
    description: str
    condition: Callable[[Any], bool]
    severity: ValidationSeverity
    fix_suggestion: Optional[str] = None
    auto_fix: Optional[Callable[[Any], Any]] = None


@dataclass
class PipelineStep:
    """파이프라인 단계"""
    name: str
    function: Callable
    dependencies: List[str] = field(default_factory=list)
    retry_count: int = 3
    timeout_seconds: int = 300
    skip_on_failure: bool = False


class FragranceDataValidator:
    """향수 데이터 전용 검증기"""

    def __init__(self):
        self.validation_rules = self._setup_validation_rules()
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.language_detector = None

        # NLP 리소스 다운로드 (필요시)
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

    def _setup_validation_rules(self) -> List[ValidationRule]:
        """검증 규칙 설정"""
        rules = []

        # 향료명 검증
        rules.append(ValidationRule(
            name="fragrance_name_valid",
            description="향료명이 유효한 형식인지 확인",
            condition=lambda x: isinstance(x, str) and len(x.strip()) >= 2 and len(x) <= 100,
            severity=ValidationSeverity.ERROR,
            fix_suggestion="향료명은 2-100자 사이의 문자열이어야 합니다.",
            auto_fix=lambda x: str(x).strip()[:100] if x else None
        ))

        # 강도 값 검증 (1-10 범위)
        rules.append(ValidationRule(
            name="intensity_range",
            description="강도 값이 1-10 범위인지 확인",
            condition=lambda x: isinstance(x, (int, float)) and 1 <= x <= 10,
            severity=ValidationSeverity.ERROR,
            fix_suggestion="강도 값은 1-10 사이여야 합니다.",
            auto_fix=lambda x: max(1, min(10, float(x))) if x is not None else None
        ))

        # 가격 검증
        rules.append(ValidationRule(
            name="price_positive",
            description="가격이 양수인지 확인",
            condition=lambda x: x is None or (isinstance(x, (int, float)) and x >= 0),
            severity=ValidationSeverity.WARNING,
            fix_suggestion="가격은 0 이상이어야 합니다.",
            auto_fix=lambda x: abs(float(x)) if x is not None else None
        ))

        # 이메일 형식 검증
        rules.append(ValidationRule(
            name="email_format",
            description="이메일 형식이 유효한지 확인",
            condition=lambda x: x is None or re.match(r'^[^@]+@[^@]+\.[^@]+$', str(x)),
            severity=ValidationSeverity.WARNING,
            fix_suggestion="올바른 이메일 형식을 사용하세요."
        ))

        # 날짜 형식 검증
        rules.append(ValidationRule(
            name="date_format",
            description="날짜 형식이 유효한지 확인",
            condition=self._is_valid_date,
            severity=ValidationSeverity.WARNING,
            fix_suggestion="올바른 날짜 형식을 사용하세요."
        ))

        # 텍스트 품질 검증
        rules.append(ValidationRule(
            name="text_quality",
            description="텍스트 품질이 좋은지 확인",
            condition=self._is_good_quality_text,
            severity=ValidationSeverity.INFO,
            fix_suggestion="텍스트 품질을 개선하세요."
        ))

        return rules

    def _is_valid_date(self, value: Any) -> bool:
        """날짜 유효성 검증"""
        if value is None:
            return True

        try:
            if isinstance(value, str):
                pd.to_datetime(value)
                return True
            elif isinstance(value, (datetime, pd.Timestamp)):
                return True
            return False
        except:
            return False

    def _is_good_quality_text(self, text: Any) -> bool:
        """텍스트 품질 검증"""
        if not isinstance(text, str) or len(text.strip()) < 5:
            return False

        # 기본 품질 체크
        if len(set(text.lower())) < 3:  # 너무 반복적
            return False

        if text.count('?') > len(text) / 10:  # 물음표 남용
            return False

        return True

    async def validate_dataframe(self, df: pd.DataFrame, schema: Dict[str, Any]) -> Tuple[pd.DataFrame, DataQualityMetrics]:
        """데이터프레임 검증 및 품질 평가"""
        logger.info(f"Validating dataframe with {len(df)} rows and {len(df.columns)} columns")

        issues_count = {}
        fixed_data = df.copy()
        total_checks = 0
        passed_checks = 0

        # 스키마 기반 검증
        for column, column_schema in schema.items():
            if column not in df.columns:
                logger.warning(f"Missing column: {column}")
                continue

            column_data = df[column]
            column_rules = column_schema.get('validation_rules', [])

            for rule_name in column_rules:
                rule = next((r for r in self.validation_rules if r.name == rule_name), None)
                if not rule:
                    continue

                total_checks += len(column_data)
                valid_mask = column_data.apply(rule.condition)
                valid_count = valid_mask.sum()
                passed_checks += valid_count

                invalid_count = len(column_data) - valid_count
                if invalid_count > 0:
                    issues_count[f"{column}_{rule_name}"] = invalid_count

                    if rule.severity == ValidationSeverity.ERROR and invalid_count > 0:
                        logger.error(f"Validation failed: {column} - {rule.description} ({invalid_count} issues)")

                    # 자동 수정 시도
                    if rule.auto_fix and invalid_count > 0:
                        try:
                            invalid_indices = ~valid_mask
                            fixed_values = column_data[invalid_indices].apply(rule.auto_fix)
                            fixed_data.loc[invalid_indices, column] = fixed_values
                            logger.info(f"Auto-fixed {invalid_count} values in {column}")
                        except Exception as e:
                            logger.warning(f"Auto-fix failed for {column}: {e}")

        # 품질 메트릭 계산
        quality_metrics = await self._calculate_quality_metrics(fixed_data, issues_count)

        return fixed_data, quality_metrics

    async def _calculate_quality_metrics(self, df: pd.DataFrame, issues_count: Dict[str, int]) -> DataQualityMetrics:
        """데이터 품질 메트릭 계산"""
        total_cells = df.size
        total_issues = sum(issues_count.values())

        # 완전성 (빈 값 비율)
        null_count = df.isnull().sum().sum()
        completeness = (total_cells - null_count) / total_cells if total_cells > 0 else 0

        # 유효성 (규칙 위반 비율)
        validity = max(0, 1 - (total_issues / total_cells)) if total_cells > 0 else 0

        # 일관성 (중복 제거)
        duplicate_count = df.duplicated().sum()
        consistency = (len(df) - duplicate_count) / len(df) if len(df) > 0 else 0

        # 고유성 (각 컬럼별 고유값 비율의 평균)
        uniqueness_scores = []
        for col in df.select_dtypes(include=['object']).columns:
            unique_ratio = df[col].nunique() / len(df) if len(df) > 0 else 0
            uniqueness_scores.append(unique_ratio)
        uniqueness = np.mean(uniqueness_scores) if uniqueness_scores else 1.0

        # 정확성 (임시로 완전성과 유효성의 평균)
        accuracy = (completeness + validity) / 2

        # 시의성 (최신성 - 날짜 컬럼 기반)
        timeliness = await self._calculate_timeliness(df)

        # 전체 점수
        weights = {
            'completeness': 0.25,
            'validity': 0.25,
            'consistency': 0.20,
            'accuracy': 0.15,
            'uniqueness': 0.10,
            'timeliness': 0.05
        }

        overall_score = (
            completeness * weights['completeness'] +
            validity * weights['validity'] +
            consistency * weights['consistency'] +
            accuracy * weights['accuracy'] +
            uniqueness * weights['uniqueness'] +
            timeliness * weights['timeliness']
        )

        # 품질 레벨 결정
        if overall_score >= 0.9:
            quality_level = DataQualityLevel.EXCELLENT
        elif overall_score >= 0.7:
            quality_level = DataQualityLevel.GOOD
        elif overall_score >= 0.5:
            quality_level = DataQualityLevel.FAIR
        elif overall_score >= 0.3:
            quality_level = DataQualityLevel.POOR
        else:
            quality_level = DataQualityLevel.CRITICAL

        # 개선 권장사항 생성
        recommendations = self._generate_recommendations(
            completeness, validity, consistency, accuracy, uniqueness, timeliness, issues_count
        )

        return DataQualityMetrics(
            completeness=completeness,
            validity=validity,
            consistency=consistency,
            accuracy=accuracy,
            uniqueness=uniqueness,
            timeliness=timeliness,
            overall_score=overall_score,
            quality_level=quality_level,
            issues_count=issues_count,
            recommendations=recommendations
        )

    async def _calculate_timeliness(self, df: pd.DataFrame) -> float:
        """시의성 계산"""
        date_columns = []

        # 날짜 형식 컬럼 찾기
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['date', 'time', 'created', 'updated']):
                try:
                    pd.to_datetime(df[col].dropna().iloc[0])
                    date_columns.append(col)
                except:
                    continue

        if not date_columns:
            return 1.0  # 날짜 컬럼이 없으면 만점

        # 가장 최근 날짜 컬럼 기준으로 시의성 계산
        timeliness_scores = []
        now = datetime.now(timezone.utc)

        for col in date_columns:
            try:
                dates = pd.to_datetime(df[col].dropna())
                if len(dates) == 0:
                    continue

                latest_date = dates.max()
                days_old = (now - latest_date.tz_localize(timezone.utc) if latest_date.tz is None else now - latest_date).days

                # 30일 이내면 1.0, 1년 이후면 0.0, 그 사이는 선형 감소
                if days_old <= 30:
                    score = 1.0
                elif days_old >= 365:
                    score = 0.0
                else:
                    score = 1.0 - (days_old - 30) / (365 - 30)

                timeliness_scores.append(score)

            except Exception as e:
                logger.warning(f"Could not calculate timeliness for column {col}: {e}")
                continue

        return np.mean(timeliness_scores) if timeliness_scores else 1.0

    def _generate_recommendations(
        self,
        completeness: float,
        validity: float,
        consistency: float,
        accuracy: float,
        uniqueness: float,
        timeliness: float,
        issues_count: Dict[str, int]
    ) -> List[str]:
        """개선 권장사항 생성"""
        recommendations = []

        if completeness < 0.8:
            recommendations.append("빈 값이 많습니다. 데이터 수집 과정을 점검하세요.")

        if validity < 0.8:
            recommendations.append("형식 규칙을 위반하는 데이터가 많습니다. 입력 검증을 강화하세요.")

        if consistency < 0.8:
            recommendations.append("중복 데이터가 많습니다. 중복 제거 과정을 추가하세요.")

        if uniqueness < 0.5:
            recommendations.append("고유값이 부족합니다. 데이터의 다양성을 확인하세요.")

        if timeliness < 0.7:
            recommendations.append("데이터가 오래되었습니다. 더 자주 업데이트하세요.")

        # 특정 이슈에 대한 권장사항
        for issue, count in issues_count.items():
            if count > 0:
                if 'email' in issue:
                    recommendations.append("이메일 형식을 확인하고 정규화하세요.")
                elif 'price' in issue:
                    recommendations.append("가격 데이터의 유효성을 검토하세요.")
                elif 'date' in issue:
                    recommendations.append("날짜 형식을 표준화하세요.")

        return list(set(recommendations))  # 중복 제거


class DataPipeline:
    """포괄적인 데이터 파이프라인"""

    def __init__(self):
        self.validator = FragranceDataValidator()
        self.steps: Dict[str, PipelineStep] = {}
        self.execution_history: List[Dict[str, Any]] = []

    def add_step(self, step: PipelineStep):
        """파이프라인 단계 추가"""
        self.steps[step.name] = step
        logger.info(f"Added pipeline step: {step.name}")

    async def extract_data(self, source: str, source_type: DataSourceType, **kwargs) -> pd.DataFrame:
        """데이터 추출 (Extract)"""
        logger.info(f"Extracting data from {source} (type: {source_type.value})")

        try:
            if source_type == DataSourceType.CSV:
                df = pd.read_csv(source, **kwargs)
            elif source_type == DataSourceType.JSON:
                df = pd.read_json(source, **kwargs)
            elif source_type == DataSourceType.JSONL:
                df = pd.read_json(source, lines=True, **kwargs)
            elif source_type == DataSourceType.EXCEL:
                df = pd.read_excel(source, **kwargs)
            elif source_type == DataSourceType.PARQUET:
                df = pd.read_parquet(source, **kwargs)
            elif source_type == DataSourceType.DATABASE:
                df = await self._extract_from_database(source, **kwargs)
            elif source_type == DataSourceType.API:
                df = await self._extract_from_api(source, **kwargs)
            else:
                raise ValueError(f"Unsupported source type: {source_type}")

            logger.info(f"Extracted {len(df)} rows and {len(df.columns)} columns")
            return df

        except Exception as e:
            logger.error(f"Data extraction failed from {source}", exception=e)
            raise DataPipelineError(f"Extraction failed: {e}")

    async def _extract_from_database(self, query: str, **kwargs) -> pd.DataFrame:
        """데이터베이스에서 데이터 추출"""
        async with get_async_session() as session:
            result = await session.execute(text(query))
            data = result.fetchall()
            columns = result.keys()
            return pd.DataFrame(data, columns=columns)

    async def _extract_from_api(self, url: str, **kwargs) -> pd.DataFrame:
        """API에서 데이터 추출"""
        async with aiohttp.ClientSession() as session:
            async with session.get(url, **kwargs) as response:
                data = await response.json()

                if isinstance(data, list):
                    return pd.DataFrame(data)
                elif isinstance(data, dict):
                    if 'data' in data:
                        return pd.DataFrame(data['data'])
                    else:
                        return pd.DataFrame([data])
                else:
                    raise ValueError("Unexpected API response format")

    async def transform_data(self, df: pd.DataFrame, transformations: List[Dict[str, Any]]) -> pd.DataFrame:
        """데이터 변환 (Transform)"""
        logger.info(f"Transforming data with {len(transformations)} transformations")

        transformed_df = df.copy()

        for transform in transformations:
            transform_type = transform.get('type')
            params = transform.get('params', {})

            try:
                if transform_type == 'drop_duplicates':
                    transformed_df = transformed_df.drop_duplicates(**params)

                elif transform_type == 'fill_null':
                    transformed_df = transformed_df.fillna(params.get('value', ''))

                elif transform_type == 'normalize_text':
                    text_columns = params.get('columns', [])
                    for col in text_columns:
                        if col in transformed_df.columns:
                            transformed_df[col] = transformed_df[col].str.strip().str.lower()

                elif transform_type == 'convert_dtype':
                    column = params.get('column')
                    dtype = params.get('dtype')
                    if column and dtype:
                        transformed_df[column] = transformed_df[column].astype(dtype)

                elif transform_type == 'add_calculated_column':
                    column_name = params.get('name')
                    formula = params.get('formula')
                    if column_name and formula:
                        transformed_df[column_name] = transformed_df.eval(formula)

                elif transform_type == 'filter_rows':
                    condition = params.get('condition')
                    if condition:
                        transformed_df = transformed_df.query(condition)

                elif transform_type == 'rename_columns':
                    mapping = params.get('mapping', {})
                    transformed_df = transformed_df.rename(columns=mapping)

                elif transform_type == 'custom':
                    func = params.get('function')
                    if func and callable(func):
                        transformed_df = func(transformed_df, **params.get('kwargs', {}))

                logger.info(f"Applied transformation: {transform_type}")

            except Exception as e:
                logger.error(f"Transformation failed: {transform_type}", exception=e)
                if not params.get('skip_on_error', False):
                    raise

        logger.info(f"Transformation completed. Result: {len(transformed_df)} rows")
        return transformed_df

    async def validate_and_clean(self, df: pd.DataFrame, schema: Dict[str, Any]) -> Tuple[pd.DataFrame, DataQualityMetrics]:
        """데이터 검증 및 정제"""
        logger.info("Starting data validation and cleaning")

        # 검증 및 자동 수정
        cleaned_df, quality_metrics = await self.validator.validate_dataframe(df, schema)

        # 품질 보고서 로깅
        logger.info("Data quality report",
                   overall_score=quality_metrics.overall_score,
                   quality_level=quality_metrics.quality_level.value,
                   completeness=quality_metrics.completeness,
                   validity=quality_metrics.validity,
                   consistency=quality_metrics.consistency)

        # 심각한 품질 문제가 있는 경우 경고
        if quality_metrics.quality_level in [DataQualityLevel.POOR, DataQualityLevel.CRITICAL]:
            logger.warning("Data quality is below acceptable threshold",
                          issues_count=quality_metrics.issues_count,
                          recommendations=quality_metrics.recommendations)

        return cleaned_df, quality_metrics

    async def load_data(self, df: pd.DataFrame, destination: str, destination_type: DataSourceType, **kwargs) -> bool:
        """데이터 적재 (Load)"""
        logger.info(f"Loading {len(df)} rows to {destination} (type: {destination_type.value})")

        try:
            if destination_type == DataSourceType.CSV:
                df.to_csv(destination, index=False, **kwargs)
            elif destination_type == DataSourceType.JSON:
                df.to_json(destination, orient='records', **kwargs)
            elif destination_type == DataSourceType.PARQUET:
                df.to_parquet(destination, **kwargs)
            elif destination_type == DataSourceType.DATABASE:
                await self._load_to_database(df, destination, **kwargs)
            else:
                raise ValueError(f"Unsupported destination type: {destination_type}")

            logger.info(f"Successfully loaded data to {destination}")
            return True

        except Exception as e:
            logger.error(f"Data loading failed to {destination}", exception=e)
            raise DataPipelineError(f"Loading failed: {e}")

    async def _load_to_database(self, df: pd.DataFrame, table_name: str, **kwargs):
        """데이터베이스에 데이터 적재"""
        from sqlalchemy import create_engine

        # 동기 엔진 사용 (pandas to_sql은 비동기 미지원)
        engine = create_engine(settings.database_url.replace('+asyncpg', ''))

        if_exists = kwargs.get('if_exists', 'append')
        df.to_sql(table_name, engine, if_exists=if_exists, index=False, **kwargs)

    async def run_full_pipeline(
        self,
        source: str,
        source_type: DataSourceType,
        destination: str,
        destination_type: DataSourceType,
        schema: Dict[str, Any],
        transformations: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """전체 파이프라인 실행"""
        pipeline_id = f"pipeline_{int(time.time())}"
        start_time = datetime.now(timezone.utc)

        logger.info(f"Starting full pipeline {pipeline_id}")

        try:
            # 1. Extract
            df = await self.extract_data(source, source_type)
            initial_rows = len(df)

            # 2. Transform
            if transformations:
                df = await self.transform_data(df, transformations)

            # 3. Validate & Clean
            df, quality_metrics = await self.validate_and_clean(df, schema)

            # 4. Load
            success = await self.load_data(df, destination, destination_type)

            # 실행 기록 저장
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            final_rows = len(df)

            execution_record = {
                "pipeline_id": pipeline_id,
                "start_time": start_time.isoformat(),
                "duration_seconds": duration,
                "source": source,
                "destination": destination,
                "initial_rows": initial_rows,
                "final_rows": final_rows,
                "rows_processed": final_rows,
                "rows_rejected": initial_rows - final_rows,
                "quality_metrics": asdict(quality_metrics),
                "success": success,
                "transformations_applied": len(transformations) if transformations else 0
            }

            self.execution_history.append(execution_record)

            logger.info(f"Pipeline {pipeline_id} completed successfully",
                       duration=duration,
                       rows_processed=final_rows,
                       quality_score=quality_metrics.overall_score)

            return execution_record

        except Exception as e:
            # 실행 실패 기록
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            execution_record = {
                "pipeline_id": pipeline_id,
                "start_time": start_time.isoformat(),
                "duration_seconds": duration,
                "source": source,
                "destination": destination,
                "success": False,
                "error": str(e)
            }

            self.execution_history.append(execution_record)

            logger.error(f"Pipeline {pipeline_id} failed", exception=e)
            raise

    def get_execution_summary(self, limit: int = 10) -> List[Dict[str, Any]]:
        """실행 이력 요약"""
        return self.execution_history[-limit:]

    async def generate_data_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """데이터 프로파일링"""
        logger.info("Generating data profile")

        profile = {
            "basic_info": {
                "rows": len(df),
                "columns": len(df.columns),
                "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
                "dtypes": df.dtypes.value_counts().to_dict()
            },
            "missing_data": {
                "total_missing": df.isnull().sum().sum(),
                "missing_percentage": (df.isnull().sum().sum() / df.size) * 100,
                "columns_with_missing": df.isnull().sum()[df.isnull().sum() > 0].to_dict()
            },
            "duplicates": {
                "duplicate_rows": df.duplicated().sum(),
                "duplicate_percentage": (df.duplicated().sum() / len(df)) * 100
            },
            "column_stats": {}
        }

        # 컬럼별 통계
        for col in df.columns:
            col_stats = {
                "dtype": str(df[col].dtype),
                "unique_values": df[col].nunique(),
                "missing_count": df[col].isnull().sum()
            }

            if df[col].dtype in ['int64', 'float64']:
                col_stats.update({
                    "mean": df[col].mean(),
                    "median": df[col].median(),
                    "std": df[col].std(),
                    "min": df[col].min(),
                    "max": df[col].max(),
                    "quartiles": {
                        "q25": df[col].quantile(0.25),
                        "q75": df[col].quantile(0.75)
                    }
                })
            elif df[col].dtype == 'object':
                col_stats.update({
                    "avg_length": df[col].str.len().mean() if df[col].dtype == 'object' else None,
                    "top_values": df[col].value_counts().head(5).to_dict()
                })

            profile["column_stats"][col] = col_stats

        return profile


# 전역 파이프라인 인스턴스
data_pipeline = DataPipeline()


# 편의 함수들
async def run_etl_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    """ETL 파이프라인 실행"""
    return await data_pipeline.run_full_pipeline(
        source=config["source"],
        source_type=DataSourceType(config["source_type"]),
        destination=config["destination"],
        destination_type=DataSourceType(config["destination_type"]),
        schema=config["schema"],
        transformations=config.get("transformations", [])
    )


async def validate_data_quality(df: pd.DataFrame, schema: Dict[str, Any]) -> DataQualityMetrics:
    """데이터 품질 검증"""
    validator = FragranceDataValidator()
    _, quality_metrics = await validator.validate_dataframe(df, schema)
    return quality_metrics


async def generate_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """데이터 품질 보고서 생성"""
    profile = await data_pipeline.generate_data_profile(df)

    # 간단한 스키마 생성 (실제로는 더 정교해야 함)
    schema = {
        col: {"validation_rules": ["fragrance_name_valid"] if "name" in col.lower() else []}
        for col in df.columns
    }

    quality_metrics = await validate_data_quality(df, schema)

    return {
        "profile": profile,
        "quality_metrics": asdict(quality_metrics),
        "generated_at": datetime.now(timezone.utc).isoformat()
    }