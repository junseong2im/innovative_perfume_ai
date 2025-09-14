"""
Fragrance AI 데이터 파이프라인 DAG
향수 데이터 수집, 전처리, 임베딩 생성 및 벡터 DB 업데이트 자동화
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import json
import logging
import pandas as pd
import requests
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.email import EmailOperator
from airflow.sensors.filesystem import FileSensor
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.models import Variable
from airflow.exceptions import AirflowSkipException
from airflow.utils.task_group import TaskGroup

# 로깅 설정
logger = logging.getLogger(__name__)

# DAG 기본 설정
default_args = {
    'owner': 'fragrance-ai',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'max_active_runs': 1,
}

# DAG 정의
dag = DAG(
    'fragrance_data_pipeline',
    default_args=default_args,
    description='향수 데이터 수집 및 처리 파이프라인',
    schedule_interval='@daily',  # 매일 실행
    catchup=False,
    tags=['fragrance', 'data-pipeline', 'ml'],
    doc_md=__doc__
)

# 설정 변수들
DATA_DIR = Variable.get("DATA_DIR", "/opt/airflow/data")
API_ENDPOINTS = Variable.get("FRAGRANCE_API_ENDPOINTS", "[]")
DB_CONN_ID = "postgres_fragrance_ai"
CHROMA_DB_PATH = Variable.get("CHROMA_DB_PATH", "/opt/airflow/data/chroma_db")

def extract_fragrance_data(**context) -> Dict[str, Any]:
    """향수 데이터 추출"""
    from fragrance_ai.utils.web_scraper import FragranceDataScraper
    from fragrance_ai.core.config import settings

    logger.info("Starting fragrance data extraction")

    try:
        # 웹 스크래핑 초기화
        scraper = FragranceDataScraper(
            rate_limit_delay=2.0,
            max_retries=3,
            enable_cache=True
        )

        # 여러 소스에서 데이터 수집
        sources = {
            'fragrantica': 'https://www.fragrantica.com/niche/',
            'basenotes': 'https://basenotes.com/fragrances/',
            'parfumo': 'https://www.parfumo.com/Perfumes/',
            # 한국 향수 쇼핑몰들
            'amorepacific': 'https://www.amorepacific.com/kr/ko/c/fragrance',
            'olive_young': 'https://www.oliveyoung.co.kr/store/display/getMCategoryList.do?dispCatNo=100000100010013'
        }

        all_data = []
        for source_name, url in sources.items():
            try:
                logger.info(f"Scraping data from {source_name}")
                data = scraper.scrape_fragrance_data(
                    url=url,
                    max_items=100,
                    include_reviews=True
                )

                # 데이터 소스 태그 추가
                for item in data:
                    item['data_source'] = source_name
                    item['scraped_at'] = datetime.utcnow().isoformat()

                all_data.extend(data)
                logger.info(f"Collected {len(data)} items from {source_name}")

            except Exception as e:
                logger.error(f"Failed to scrape {source_name}: {e}")
                continue

        # 데이터 저장
        output_file = Path(DATA_DIR) / f"raw_fragrance_data_{context['ds']}.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Extracted {len(all_data)} fragrance items")
        return {
            'total_items': len(all_data),
            'output_file': str(output_file),
            'sources': list(sources.keys())
        }

    except Exception as e:
        logger.error(f"Data extraction failed: {e}")
        raise

def validate_and_clean_data(**context) -> Dict[str, Any]:
    """데이터 검증 및 정제"""
    from fragrance_ai.utils.data_cleaner import FragranceDataCleaner

    logger.info("Starting data validation and cleaning")

    try:
        # 이전 태스크에서 생성된 파일 로드
        ti = context['ti']
        extract_result = ti.xcom_pull(task_ids='extract_fragrance_data')
        input_file = extract_result['output_file']

        with open(input_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        # 데이터 정제기 초기화
        cleaner = FragranceDataCleaner(
            remove_duplicates=True,
            validate_schema=True,
            normalize_text=True
        )

        # 데이터 정제 수행
        cleaned_data = cleaner.clean_fragrance_data(raw_data)

        # 품질 메트릭 계산
        quality_metrics = cleaner.calculate_quality_metrics(cleaned_data)

        # 정제된 데이터 저장
        output_file = Path(DATA_DIR) / f"cleaned_fragrance_data_{context['ds']}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

        # 품질 리포트 저장
        quality_report_file = Path(DATA_DIR) / f"quality_report_{context['ds']}.json"
        with open(quality_report_file, 'w', encoding='utf-8') as f:
            json.dump(quality_metrics, f, ensure_ascii=False, indent=2)

        logger.info(f"Cleaned data: {len(cleaned_data)} items")
        logger.info(f"Quality score: {quality_metrics.get('overall_score', 0):.2f}")

        return {
            'cleaned_items': len(cleaned_data),
            'output_file': str(output_file),
            'quality_metrics': quality_metrics
        }

    except Exception as e:
        logger.error(f"Data cleaning failed: {e}")
        raise

def generate_embeddings(**context) -> Dict[str, Any]:
    """임베딩 생성"""
    from fragrance_ai.models.embedding import AdvancedKoreanFragranceEmbedding
    import numpy as np

    logger.info("Starting embedding generation")

    try:
        # 이전 태스크에서 정제된 데이터 로드
        ti = context['ti']
        clean_result = ti.xcom_pull(task_ids='validate_and_clean_data')
        input_file = clean_result['output_file']

        with open(input_file, 'r', encoding='utf-8') as f:
            cleaned_data = json.load(f)

        # 임베딩 모델 초기화
        embedding_model = AdvancedKoreanFragranceEmbedding(
            use_adapter=True,
            enable_multi_aspect=True,
            cache_size=5000
        )

        # 다양한 측면의 임베딩 생성
        embeddings_data = {
            'basic': [],
            'multi_aspect': [],
            'metadata': []
        }

        batch_size = 32
        for i in range(0, len(cleaned_data), batch_size):
            batch = cleaned_data[i:i + batch_size]

            # 기본 텍스트 준비
            texts = []
            for item in batch:
                # 향수 프로필 텍스트 조합
                profile_text = f"{item.get('name', '')} {item.get('brand', '')} "
                profile_text += f"톱노트: {', '.join(item.get('top_notes', []))} "
                profile_text += f"미들노트: {', '.join(item.get('heart_notes', []))} "
                profile_text += f"베이스노트: {', '.join(item.get('base_notes', []))} "
                profile_text += f"설명: {item.get('description', '')}"
                texts.append(profile_text.strip())

            # 기본 임베딩 생성
            basic_embeddings = embedding_model._encode_batch(texts)
            embeddings_data['basic'].extend(basic_embeddings.tolist())

            # 다면적 임베딩 생성
            multi_aspect_embeddings = embedding_model.encode_multi_aspect(texts)
            embeddings_data['multi_aspect'].extend([
                {aspect: emb.tolist() for aspect, emb in aspects.items()}
                for aspects in multi_aspect_embeddings
            ])

            # 메타데이터 저장
            embeddings_data['metadata'].extend([
                {
                    'id': item.get('id', f"item_{i+j}"),
                    'name': item.get('name'),
                    'brand': item.get('brand'),
                    'data_source': item.get('data_source')
                }
                for j, item in enumerate(batch)
            ])

            logger.info(f"Generated embeddings for batch {i//batch_size + 1}")

        # 임베딩 데이터 저장
        embeddings_file = Path(DATA_DIR) / f"fragrance_embeddings_{context['ds']}.json"
        with open(embeddings_file, 'w', encoding='utf-8') as f:
            json.dump(embeddings_data, f, ensure_ascii=False, indent=2)

        # NumPy 형식으로도 저장 (빠른 로딩용)
        np_embeddings_file = Path(DATA_DIR) / f"fragrance_embeddings_{context['ds']}.npz"
        np.savez_compressed(
            np_embeddings_file,
            basic_embeddings=np.array(embeddings_data['basic']),
            metadata=embeddings_data['metadata']
        )

        logger.info(f"Generated embeddings for {len(embeddings_data['basic'])} items")

        return {
            'embedding_count': len(embeddings_data['basic']),
            'embeddings_file': str(embeddings_file),
            'np_embeddings_file': str(np_embeddings_file)
        }

    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise

def update_vector_database(**context) -> Dict[str, Any]:
    """벡터 데이터베이스 업데이트"""
    from fragrance_ai.core.vector_store import VectorStore
    import numpy as np

    logger.info("Starting vector database update")

    try:
        # 이전 태스크에서 생성된 임베딩 로드
        ti = context['ti']
        embedding_result = ti.xcom_pull(task_ids='generate_embeddings')
        embeddings_file = embedding_result['embeddings_file']

        with open(embeddings_file, 'r', encoding='utf-8') as f:
            embeddings_data = json.load(f)

        # 벡터 스토어 초기화
        vector_store = VectorStore()

        # 컬렉션별 데이터 분류 및 추가
        collections_updated = {}

        # 1. 향수 노트 컬렉션 업데이트
        fragrance_notes_data = []
        for i, metadata in enumerate(embeddings_data['metadata']):
            fragrance_notes_data.append({
                'id': f"fragrance_{metadata['id']}",
                'embedding': embeddings_data['basic'][i],
                'metadata': metadata
            })

        vector_store.batch_add_fragrance_notes(
            collection_name="fragrance_notes",
            notes_data=fragrance_notes_data
        )
        collections_updated['fragrance_notes'] = len(fragrance_notes_data)

        # 2. 레시피 컬렉션 업데이트 (다면적 임베딩 사용)
        recipes_data = []
        for i, metadata in enumerate(embeddings_data['metadata']):
            if embeddings_data['multi_aspect'][i].get('scent_profile'):
                recipes_data.append({
                    'id': f"recipe_{metadata['id']}",
                    'embedding': embeddings_data['multi_aspect'][i]['scent_profile'],
                    'metadata': {**metadata, 'type': 'recipe'}
                })

        if recipes_data:
            vector_store.batch_add_documents(
                collection_name="recipes",
                documents=recipes_data
            )
            collections_updated['recipes'] = len(recipes_data)

        # 3. 무드 디스크립션 컬렉션 업데이트
        mood_data = []
        for i, metadata in enumerate(embeddings_data['metadata']):
            if embeddings_data['multi_aspect'][i].get('mood_emotion'):
                mood_data.append({
                    'id': f"mood_{metadata['id']}",
                    'embedding': embeddings_data['multi_aspect'][i]['mood_emotion'],
                    'metadata': {**metadata, 'type': 'mood'}
                })

        if mood_data:
            vector_store.batch_add_documents(
                collection_name="mood_descriptions",
                documents=mood_data
            )
            collections_updated['mood_descriptions'] = len(mood_data)

        # 통계 정보 수집
        total_updated = sum(collections_updated.values())

        logger.info(f"Updated vector database: {total_updated} total items")
        for collection, count in collections_updated.items():
            logger.info(f"  {collection}: {count} items")

        return {
            'total_updated': total_updated,
            'collections_updated': collections_updated,
            'vector_db_path': CHROMA_DB_PATH
        }

    except Exception as e:
        logger.error(f"Vector database update failed: {e}")
        raise

def update_postgresql_database(**context) -> Dict[str, Any]:
    """PostgreSQL 데이터베이스 업데이트"""
    logger.info("Starting PostgreSQL database update")

    try:
        # 이전 태스크에서 정제된 데이터 로드
        ti = context['ti']
        clean_result = ti.xcom_pull(task_ids='validate_and_clean_data')
        input_file = clean_result['output_file']

        with open(input_file, 'r', encoding='utf-8') as f:
            cleaned_data = json.load(f)

        # PostgreSQL 연결
        postgres_hook = PostgresHook(postgres_conn_id=DB_CONN_ID)

        # 배치 삽입을 위한 데이터 준비
        fragrance_records = []
        recipe_records = []

        for item in cleaned_data:
            # 향수 기본 정보
            fragrance_record = (
                item.get('id'),
                item.get('name'),
                item.get('brand'),
                item.get('description'),
                json.dumps(item.get('top_notes', [])),
                json.dumps(item.get('heart_notes', [])),
                json.dumps(item.get('base_notes', [])),
                item.get('price'),
                item.get('rating'),
                item.get('data_source'),
                datetime.utcnow()
            )
            fragrance_records.append(fragrance_record)

            # 레시피 정보 (있는 경우)
            if item.get('recipe'):
                recipe_record = (
                    f"recipe_{item.get('id')}",
                    item.get('id'),
                    json.dumps(item['recipe']),
                    item['recipe'].get('type', 'basic'),
                    datetime.utcnow()
                )
                recipe_records.append(recipe_record)

        # 배치 삽입 실행
        fragrance_insert_sql = """
            INSERT INTO fragrance_notes (
                id, name, brand, description, top_notes, heart_notes,
                base_notes, price, rating, data_source, created_at
            ) VALUES %s
            ON CONFLICT (id) DO UPDATE SET
                name = EXCLUDED.name,
                brand = EXCLUDED.brand,
                description = EXCLUDED.description,
                top_notes = EXCLUDED.top_notes,
                heart_notes = EXCLUDED.heart_notes,
                base_notes = EXCLUDED.base_notes,
                price = EXCLUDED.price,
                rating = EXCLUDED.rating,
                updated_at = CURRENT_TIMESTAMP;
        """

        if fragrance_records:
            postgres_hook.run(fragrance_insert_sql, parameters=fragrance_records)

        recipe_insert_sql = """
            INSERT INTO recipes (id, fragrance_id, content, recipe_type, created_at)
            VALUES %s
            ON CONFLICT (id) DO UPDATE SET
                content = EXCLUDED.content,
                recipe_type = EXCLUDED.recipe_type,
                updated_at = CURRENT_TIMESTAMP;
        """

        if recipe_records:
            postgres_hook.run(recipe_insert_sql, parameters=recipe_records)

        logger.info(f"Updated PostgreSQL: {len(fragrance_records)} fragrances, {len(recipe_records)} recipes")

        return {
            'fragrances_updated': len(fragrance_records),
            'recipes_updated': len(recipe_records)
        }

    except Exception as e:
        logger.error(f"PostgreSQL update failed: {e}")
        raise

def validate_pipeline_results(**context) -> Dict[str, Any]:
    """파이프라인 결과 검증"""
    logger.info("Starting pipeline validation")

    try:
        ti = context['ti']

        # 모든 이전 태스크 결과 수집
        extract_result = ti.xcom_pull(task_ids='extract_fragrance_data')
        clean_result = ti.xcom_pull(task_ids='validate_and_clean_data')
        embedding_result = ti.xcom_pull(task_ids='generate_embeddings')
        vector_result = ti.xcom_pull(task_ids='update_vector_database')
        postgres_result = ti.xcom_pull(task_ids='update_postgresql_database')

        # 검증 메트릭 계산
        validation_results = {
            'extraction': {
                'total_extracted': extract_result.get('total_items', 0),
                'sources_used': len(extract_result.get('sources', [])),
                'success': extract_result.get('total_items', 0) > 0
            },
            'cleaning': {
                'items_cleaned': clean_result.get('cleaned_items', 0),
                'quality_score': clean_result.get('quality_metrics', {}).get('overall_score', 0),
                'success': clean_result.get('quality_metrics', {}).get('overall_score', 0) > 0.7
            },
            'embeddings': {
                'embeddings_generated': embedding_result.get('embedding_count', 0),
                'success': embedding_result.get('embedding_count', 0) > 0
            },
            'vector_db': {
                'items_updated': vector_result.get('total_updated', 0),
                'collections': vector_result.get('collections_updated', {}),
                'success': vector_result.get('total_updated', 0) > 0
            },
            'postgres_db': {
                'fragrances_updated': postgres_result.get('fragrances_updated', 0),
                'recipes_updated': postgres_result.get('recipes_updated', 0),
                'success': postgres_result.get('fragrances_updated', 0) > 0
            }
        }

        # 전체 파이프라인 성공 여부 판단
        all_success = all(
            stage_result['success']
            for stage_result in validation_results.values()
        )

        validation_results['pipeline_success'] = all_success
        validation_results['validation_timestamp'] = datetime.utcnow().isoformat()

        # 검증 리포트 저장
        validation_file = Path(DATA_DIR) / f"pipeline_validation_{context['ds']}.json"
        with open(validation_file, 'w', encoding='utf-8') as f:
            json.dump(validation_results, f, ensure_ascii=False, indent=2)

        logger.info(f"Pipeline validation completed. Success: {all_success}")

        if not all_success:
            raise ValueError("Pipeline validation failed")

        return validation_results

    except Exception as e:
        logger.error(f"Pipeline validation failed: {e}")
        raise

def send_completion_notification(**context) -> None:
    """완료 알림 발송"""
    logger.info("Sending completion notification")

    try:
        ti = context['ti']
        validation_result = ti.xcom_pull(task_ids='validate_pipeline_results')

        # 알림 메시지 구성
        success = validation_result.get('pipeline_success', False)
        status = "SUCCESS" if success else "FAILED"

        message = f"""
        향수 데이터 파이프라인 실행 완료

        실행 날짜: {context['ds']}
        상태: {status}

        실행 결과:
        - 추출된 데이터: {validation_result.get('extraction', {}).get('total_extracted', 0)} 항목
        - 정제된 데이터: {validation_result.get('cleaning', {}).get('items_cleaned', 0)} 항목
        - 생성된 임베딩: {validation_result.get('embeddings', {}).get('embeddings_generated', 0)} 개
        - 벡터 DB 업데이트: {validation_result.get('vector_db', {}).get('items_updated', 0)} 항목
        - PostgreSQL 업데이트: {validation_result.get('postgres_db', {}).get('fragrances_updated', 0)} 향수

        품질 점수: {validation_result.get('cleaning', {}).get('quality_score', 0):.2f}
        """

        # Slack 또는 이메일 알림 (설정에 따라)
        logger.info(f"Pipeline completion notification: {message}")

    except Exception as e:
        logger.error(f"Failed to send notification: {e}")

# DAG 태스크 정의
with dag:

    # 1. 데이터 추출
    extract_task = PythonOperator(
        task_id='extract_fragrance_data',
        python_callable=extract_fragrance_data,
        doc_md="향수 데이터를 다양한 소스에서 추출"
    )

    # 2. 데이터 검증 및 정제
    clean_task = PythonOperator(
        task_id='validate_and_clean_data',
        python_callable=validate_and_clean_data,
        doc_md="추출된 데이터를 검증하고 정제"
    )

    # 3. 임베딩 생성
    embedding_task = PythonOperator(
        task_id='generate_embeddings',
        python_callable=generate_embeddings,
        doc_md="정제된 데이터에서 임베딩 생성"
    )

    # 4. 데이터베이스 업데이트 (병렬)
    with TaskGroup("update_databases") as db_update_group:

        vector_db_task = PythonOperator(
            task_id='update_vector_database',
            python_callable=update_vector_database,
            doc_md="벡터 데이터베이스 업데이트"
        )

        postgres_task = PythonOperator(
            task_id='update_postgresql_database',
            python_callable=update_postgresql_database,
            doc_md="PostgreSQL 데이터베이스 업데이트"
        )

    # 5. 파이프라인 검증
    validation_task = PythonOperator(
        task_id='validate_pipeline_results',
        python_callable=validate_pipeline_results,
        doc_md="파이프라인 실행 결과 검증"
    )

    # 6. 완료 알림
    notification_task = PythonOperator(
        task_id='send_completion_notification',
        python_callable=send_completion_notification,
        doc_md="파이프라인 완료 알림 발송",
        trigger_rule='all_done'  # 실패해도 실행
    )

    # 7. 데이터 정리 (선택적)
    cleanup_task = BashOperator(
        task_id='cleanup_temp_files',
        bash_command=f'find {DATA_DIR} -name "*{{"{{ ds }}"}}*" -type f -mtime +7 -delete',
        doc_md="7일 이전 임시 파일 정리",
        trigger_rule='all_success'
    )

# 태스크 의존성 설정
extract_task >> clean_task >> embedding_task >> db_update_group
db_update_group >> validation_task >> [notification_task, cleanup_task]