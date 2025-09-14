#!/usr/bin/env python3
"""
실제 향수 데이터셋 구축 스크립트
고품질 향수 데이터를 수집하고 임베딩을 생성하여 시스템에 로드
"""

import asyncio
import logging
import json
import csv
from typing import Dict, List, Any, Optional
from pathlib import Path
import argparse
from datetime import datetime
import pandas as pd

# 프로젝트 모듈
from fragrance_ai.utils.web_scraper import FragranceDataScraper
from fragrance_ai.utils.data_cleaner import FragranceDataCleaner
from fragrance_ai.models.embedding import AdvancedKoreanFragranceEmbedding
from fragrance_ai.core.vector_store import VectorStore
from fragrance_ai.database.connection import get_db_session
from fragrance_ai.database.models import FragranceNote, Recipe
from fragrance_ai.core.config import settings

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FragranceDatasetBuilder:
    """향수 데이터셋 구축기"""

    def __init__(
        self,
        data_dir: str = "./data",
        enable_web_scraping: bool = True,
        use_sample_data: bool = False
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.enable_web_scraping = enable_web_scraping
        self.use_sample_data = use_sample_data

        # 컴포넌트 초기화
        self.scraper = FragranceDataScraper() if enable_web_scraping else None
        self.cleaner = FragranceDataCleaner()
        self.embedding_model = AdvancedKoreanFragranceEmbedding()
        self.vector_store = VectorStore()

        logger.info("FragranceDatasetBuilder initialized")

    async def build_complete_dataset(self) -> Dict[str, Any]:
        """완전한 향수 데이터셋 구축"""

        logger.info("Starting complete dataset build process")

        stats = {
            'start_time': datetime.utcnow(),
            'raw_items_collected': 0,
            'cleaned_items': 0,
            'embeddings_generated': 0,
            'vector_db_items': 0,
            'database_items': 0,
            'quality_score': 0.0
        }

        try:
            # 1. 데이터 수집
            if self.use_sample_data:
                raw_data = self._generate_sample_data()
            else:
                raw_data = await self._collect_fragrance_data()

            stats['raw_items_collected'] = len(raw_data)
            logger.info(f"Collected {len(raw_data)} raw items")

            # 2. 데이터 정제
            cleaned_data = self.cleaner.clean_fragrance_data(raw_data)
            stats['cleaned_items'] = len(cleaned_data)

            quality_metrics = self.cleaner.calculate_quality_metrics(cleaned_data)
            stats['quality_score'] = quality_metrics.overall_score

            logger.info(f"Cleaned data: {len(cleaned_data)} items, Quality: {quality_metrics.overall_score:.2f}")

            # 3. 임베딩 생성
            embeddings_data = await self._generate_embeddings(cleaned_data)
            stats['embeddings_generated'] = len(embeddings_data['basic'])

            # 4. 벡터 데이터베이스 업데이트
            vector_stats = await self._update_vector_database(cleaned_data, embeddings_data)
            stats['vector_db_items'] = vector_stats['total_items']

            # 5. PostgreSQL 데이터베이스 업데이트
            db_stats = await self._update_postgresql_database(cleaned_data)
            stats['database_items'] = db_stats['items_saved']

            # 6. 데이터셋 메타데이터 저장
            await self._save_dataset_metadata(stats, quality_metrics)

            stats['end_time'] = datetime.utcnow()
            stats['total_duration'] = (stats['end_time'] - stats['start_time']).total_seconds()

            logger.info(f"Dataset build completed successfully in {stats['total_duration']:.2f} seconds")
            return stats

        except Exception as e:
            logger.error(f"Dataset build failed: {e}")
            raise

    async def _collect_fragrance_data(self) -> List[Dict[str, Any]]:
        """웹에서 향수 데이터 수집"""

        logger.info("Collecting fragrance data from web sources")

        if not self.scraper:
            raise ValueError("Web scraping is disabled")

        all_data = []

        # 데이터 소스 정의
        data_sources = [
            {
                'name': 'fragrantica_niche',
                'url': 'https://www.fragrantica.com/niche/',
                'max_items': 200
            },
            {
                'name': 'fragrantica_designer',
                'url': 'https://www.fragrantica.com/designers/',
                'max_items': 200
            },
            {
                'name': 'olive_young',
                'url': 'https://www.oliveyoung.co.kr/store/display/getMCategoryList.do?dispCatNo=100000100010013',
                'max_items': 100
            },
            {
                'name': 'amore_pacific',
                'url': 'https://www.amorepacific.com/kr/ko/c/fragrance',
                'max_items': 50
            }
        ]

        # 각 소스에서 데이터 수집
        for source in data_sources:
            try:
                logger.info(f"Scraping {source['name']}")

                source_data = self.scraper.scrape_fragrance_data(
                    url=source['url'],
                    max_items=source['max_items'],
                    include_reviews=True
                )

                # 소스 태그 추가
                for item in source_data:
                    item['data_source'] = source['name']

                all_data.extend(source_data)
                logger.info(f"Collected {len(source_data)} items from {source['name']}")

                # 요청 간 딜레이
                await asyncio.sleep(2)

            except Exception as e:
                logger.error(f"Failed to scrape {source['name']}: {e}")
                continue

        # 데이터 저장
        raw_data_file = self.data_dir / f"raw_fragrance_data_{datetime.utcnow().strftime('%Y%m%d')}.json"
        with open(raw_data_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)

        return all_data

    def _generate_sample_data(self) -> List[Dict[str, Any]]:
        """샘플 향수 데이터 생성 (테스트용)"""

        logger.info("Generating sample fragrance data")

        sample_data = [
            {
                'id': 'sample_001',
                'name': '블랙 오키드',
                'brand': 'Tom Ford',
                'description': '신비롭고 관능적인 플로럴 오리엔탈 향수. 블랙 트뤼플과 일랑일랑의 절묘한 조합이 독특한 매력을 선사합니다.',
                'top_notes': ['트뤼플', '일랑일랑', '베르가못', '블랙커런트'],
                'heart_notes': ['오키드', '프루잇 노트', '스파이시 노트'],
                'base_notes': ['패츌리', '바닐라', '인센스', '샌달우드'],
                'price': 150000.0,
                'rating': 4.3,
                'gender': 'women',
                'season': ['가을', '겨울'],
                'data_source': 'sample'
            },
            {
                'id': 'sample_002',
                'name': '사바주',
                'brand': 'Dior',
                'description': '프레시하고 스파이시한 아로마틱 향수. 베르가못과 시치리안 만다린의 상쾌함과 암브록산의 깊이가 어우러집니다.',
                'top_notes': ['베르가못', '시치리안 만다린'],
                'heart_notes': ['사이클라멘 나뭇잎', '제라늄', '라벤더', '시치리안 엘레미'],
                'base_notes': ['암브록산', '바닐라'],
                'price': 120000.0,
                'rating': 4.5,
                'gender': 'men',
                'season': ['봄', '여름'],
                'data_source': 'sample'
            },
            {
                'id': 'sample_003',
                'name': '라이트 블루',
                'brand': 'Dolce & Gabbana',
                'description': '지중해의 여름을 담은 프레시한 시트러스 향수. 시칠리아 레몬과 그래니 스미스 애플의 상큼함이 인상적입니다.',
                'top_notes': ['시칠리아 레몬', '애플', '시더'],
                'heart_notes': ['라벤더', '화이트 로즈', '로즈마리'],
                'base_notes': ['시더우드', '오크모스', '머스크'],
                'price': 80000.0,
                'rating': 4.1,
                'gender': 'women',
                'season': ['봄', '여름'],
                'data_source': 'sample'
            },
            {
                'id': 'sample_004',
                'name': '미스 디올',
                'brand': 'Dior',
                'description': '로맨틱하고 우아한 플로럴 부케. 이탈리안 만다린과 불가리안 로즈의 조화가 여성스러운 매력을 완성합니다.',
                'top_notes': ['이탈리안 만다린', '핑크 페퍼'],
                'heart_notes': ['불가리안 로즈', '피오니', '릴리 오브 더 밸리'],
                'base_notes': ['인도네시안 패츌리', '로즈우드'],
                'price': 130000.0,
                'rating': 4.4,
                'gender': 'women',
                'season': ['봄', '사계절'],
                'data_source': 'sample'
            },
            {
                'id': 'sample_005',
                'name': '라이브 이레지스터블리',
                'brand': 'Givenchy',
                'description': '매혹적이고 강렬한 플로럴 프루티 향수. 로즈와 리치의 달콤함이 패츌리의 깊이와 만나 잊을 수 없는 향을 만듭니다.',
                'top_notes': ['멀베리', '리치', '블랙베리'],
                'heart_notes': ['프렌치 로즈', '피오니'],
                'base_notes': ['캐시미어우드', '패츌리'],
                'price': 95000.0,
                'rating': 4.2,
                'gender': 'women',
                'season': ['가을', '겨울'],
                'data_source': 'sample'
            },
            # 한국 브랜드 샘플
            {
                'id': 'sample_006',
                'name': '설화수 윤조에센스',
                'brand': '설화수',
                'description': '동양적이고 우아한 한방 향수. 자음단과 궁중 비밀 처방을 현대적으로 재해석한 고급스러운 향입니다.',
                'top_notes': ['한라봉', '녹차', '생강'],
                'heart_notes': ['모란', '연꽃', '국화'],
                'base_notes': ['인삼', '감초', '백단향'],
                'price': 180000.0,
                'rating': 4.6,
                'gender': 'unisex',
                'season': ['사계절'],
                'data_source': 'sample'
            },
            {
                'id': 'sample_007',
                'name': '라네즈 퍼퓸',
                'brand': '라네즈',
                'description': '수분과 생기를 담은 프레시 플로럴 향수. 미네랄 워터와 워터 라일리의 청량감이 특징입니다.',
                'top_notes': ['워터 드롭', '한라봉', '아쿠아 민트'],
                'heart_notes': ['워터 라일리', '하이드랜지아', '프리지아'],
                'base_notes': ['아쿠아 머스크', '시더우드'],
                'price': 60000.0,
                'rating': 4.0,
                'gender': 'women',
                'season': ['봄', '여름'],
                'data_source': 'sample'
            }
        ]

        # 더 많은 샘플 생성 (변형)
        extended_data = []
        for base_item in sample_data:
            extended_data.append(base_item)

            # 각 기본 아이템에서 2-3개의 변형 생성
            for i in range(2):
                variant = base_item.copy()
                variant['id'] = f"{base_item['id']}_var_{i+1}"
                variant['name'] = f"{base_item['name']} {['Intense', 'Light', 'Summer', 'Winter'][i]}"

                # 가격 변경
                variant['price'] = base_item['price'] * (0.8 + i * 0.2)

                # 평점 약간 변경
                variant['rating'] = min(5.0, max(3.0, base_item['rating'] + (i-1) * 0.1))

                extended_data.append(variant)

        logger.info(f"Generated {len(extended_data)} sample fragrance items")
        return extended_data

    async def _generate_embeddings(self, cleaned_data: List[Dict[str, Any]]) -> Dict[str, List]:
        """임베딩 생성"""

        logger.info(f"Generating embeddings for {len(cleaned_data)} items")

        embeddings_data = {
            'basic': [],
            'multi_aspect': [],
            'metadata': []
        }

        batch_size = 32
        for i in range(0, len(cleaned_data), batch_size):
            batch = cleaned_data[i:i + batch_size]

            # 텍스트 준비
            texts = []
            for item in batch:
                # 향수 프로필 텍스트 생성
                profile_parts = [
                    item.get('name', ''),
                    item.get('brand', ''),
                    item.get('description', '')
                ]

                # 노트 정보 추가
                if item.get('top_notes'):
                    profile_parts.append(f"톱노트: {', '.join(item['top_notes'])}")
                if item.get('heart_notes'):
                    profile_parts.append(f"미들노트: {', '.join(item['heart_notes'])}")
                if item.get('base_notes'):
                    profile_parts.append(f"베이스노트: {', '.join(item['base_notes'])}")

                profile_text = ' '.join(filter(None, profile_parts))
                texts.append(profile_text)

            # 기본 임베딩 생성
            basic_embeddings_result = await self.embedding_model.encode_async(texts)
            embeddings_data['basic'].extend(basic_embeddings_result.embeddings.tolist())

            # 다면적 임베딩 생성
            multi_aspect_embeddings = self.embedding_model.encode_multi_aspect(texts)
            embeddings_data['multi_aspect'].extend([
                {aspect: emb.tolist() for aspect, emb in multi_aspect_embeddings.items()}
            ])

            # 메타데이터 저장
            for item in batch:
                embeddings_data['metadata'].append({
                    'id': item.get('id'),
                    'name': item.get('name'),
                    'brand': item.get('brand'),
                    'data_source': item.get('data_source')
                })

            logger.info(f"Generated embeddings for batch {i//batch_size + 1}/{(len(cleaned_data)-1)//batch_size + 1}")

        # 임베딩 데이터 저장
        embeddings_file = self.data_dir / f"fragrance_embeddings_{datetime.utcnow().strftime('%Y%m%d')}.json"
        with open(embeddings_file, 'w', encoding='utf-8') as f:
            json.dump(embeddings_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved embeddings to {embeddings_file}")
        return embeddings_data

    async def _update_vector_database(
        self,
        cleaned_data: List[Dict[str, Any]],
        embeddings_data: Dict[str, List]
    ) -> Dict[str, Any]:
        """벡터 데이터베이스 업데이트"""

        logger.info("Updating vector database")

        stats = {'collections': {}, 'total_items': 0}

        try:
            # 1. 향수 노트 컬렉션
            fragrance_notes_data = []
            for i, item in enumerate(cleaned_data):
                if i < len(embeddings_data['basic']):
                    fragrance_notes_data.append({
                        'id': f"fragrance_{item['id']}",
                        'embedding': embeddings_data['basic'][i],
                        'metadata': {
                            'name': item['name'],
                            'brand': item['brand'],
                            'description': item.get('description', ''),
                            'top_notes': item.get('top_notes', []),
                            'heart_notes': item.get('heart_notes', []),
                            'base_notes': item.get('base_notes', []),
                            'price': item.get('price'),
                            'rating': item.get('rating'),
                            'gender': item.get('gender'),
                            'season': item.get('season', []),
                            'data_source': item.get('data_source')
                        }
                    })

            self.vector_store.batch_add_fragrance_notes(
                collection_name="fragrance_notes",
                notes_data=fragrance_notes_data
            )
            stats['collections']['fragrance_notes'] = len(fragrance_notes_data)

            # 2. 레시피 컬렉션 (다면적 임베딩 사용)
            recipes_data = []
            for i, item in enumerate(cleaned_data):
                if (i < len(embeddings_data['multi_aspect']) and
                    'scent_profile' in embeddings_data['multi_aspect'][i]):

                    recipes_data.append({
                        'id': f"recipe_{item['id']}",
                        'embedding': embeddings_data['multi_aspect'][i]['scent_profile'],
                        'metadata': {
                            **fragrance_notes_data[i]['metadata'],
                            'type': 'recipe'
                        }
                    })

            if recipes_data:
                self.vector_store.batch_add_documents(
                    collection_name="recipes",
                    documents=recipes_data
                )
                stats['collections']['recipes'] = len(recipes_data)

            # 3. 무드 디스크립션 컬렉션
            mood_data = []
            for i, item in enumerate(cleaned_data):
                if (i < len(embeddings_data['multi_aspect']) and
                    'mood_emotion' in embeddings_data['multi_aspect'][i]):

                    mood_data.append({
                        'id': f"mood_{item['id']}",
                        'embedding': embeddings_data['multi_aspect'][i]['mood_emotion'],
                        'metadata': {
                            **fragrance_notes_data[i]['metadata'],
                            'type': 'mood'
                        }
                    })

            if mood_data:
                self.vector_store.batch_add_documents(
                    collection_name="mood_descriptions",
                    documents=mood_data
                )
                stats['collections']['mood_descriptions'] = len(mood_data)

            stats['total_items'] = sum(stats['collections'].values())

            logger.info(f"Vector database updated: {stats['total_items']} total items")
            return stats

        except Exception as e:
            logger.error(f"Vector database update failed: {e}")
            raise

    async def _update_postgresql_database(self, cleaned_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """PostgreSQL 데이터베이스 업데이트"""

        logger.info("Updating PostgreSQL database")

        stats = {'items_saved': 0, 'errors': 0}

        try:
            async for session in get_db_session():
                for item in cleaned_data:
                    try:
                        # 향수 노트 저장
                        fragrance_note = FragranceNote(
                            id=item['id'],
                            name=item['name'],
                            brand=item['brand'],
                            description=item.get('description', ''),
                            top_notes=item.get('top_notes', []),
                            heart_notes=item.get('heart_notes', []),
                            base_notes=item.get('base_notes', []),
                            price=item.get('price'),
                            rating=item.get('rating'),
                            gender=item.get('gender'),
                            season=item.get('season', []),
                            data_source=item.get('data_source')
                        )

                        # 중복 체크 및 업데이트
                        existing = await session.get(FragranceNote, item['id'])
                        if existing:
                            # 기존 레코드 업데이트
                            for field, value in item.items():
                                if hasattr(existing, field) and value is not None:
                                    setattr(existing, field, value)
                        else:
                            session.add(fragrance_note)

                        stats['items_saved'] += 1

                    except Exception as e:
                        logger.error(f"Failed to save item {item.get('id', 'unknown')}: {e}")
                        stats['errors'] += 1
                        continue

                await session.commit()

            logger.info(f"PostgreSQL database updated: {stats['items_saved']} items saved, {stats['errors']} errors")
            return stats

        except Exception as e:
            logger.error(f"PostgreSQL database update failed: {e}")
            raise

    async def _save_dataset_metadata(
        self,
        stats: Dict[str, Any],
        quality_metrics: Any
    ) -> None:
        """데이터셋 메타데이터 저장"""

        metadata = {
            'dataset_id': f"fragrance_dataset_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            'created_at': datetime.utcnow().isoformat(),
            'statistics': stats,
            'quality_metrics': {
                'overall_score': quality_metrics.overall_score,
                'total_items': quality_metrics.total_items,
                'field_completeness': quality_metrics.field_completeness,
                'data_consistency': quality_metrics.data_consistency
            },
            'configuration': {
                'embedding_model': self.embedding_model.model_name,
                'vector_dimension': getattr(self.embedding_model, 'vector_dimension', 384),
                'data_sources': ['fragrantica', 'olive_young', 'amore_pacific', 'sample'],
                'processing_pipeline': [
                    'data_collection',
                    'data_cleaning',
                    'embedding_generation',
                    'vector_database_update',
                    'postgresql_update'
                ]
            }
        }

        # 메타데이터 저장
        metadata_file = self.data_dir / f"dataset_metadata_{datetime.utcnow().strftime('%Y%m%d')}.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logger.info(f"Dataset metadata saved to {metadata_file}")

    async def export_dataset_for_analysis(self) -> str:
        """분석용 데이터셋 내보내기"""

        logger.info("Exporting dataset for analysis")

        try:
            # 데이터베이스에서 데이터 로드
            async for session in get_db_session():
                result = await session.execute("SELECT * FROM fragrance_notes")
                fragrance_data = result.fetchall()

            if not fragrance_data:
                raise ValueError("No data found in database")

            # DataFrame 생성
            df = pd.DataFrame(fragrance_data)

            # CSV 내보내기
            csv_file = self.data_dir / f"fragrance_dataset_analysis_{datetime.utcnow().strftime('%Y%m%d')}.csv"
            df.to_csv(csv_file, index=False, encoding='utf-8')

            # 기본 통계 생성
            stats_file = self.data_dir / f"dataset_stats_{datetime.utcnow().strftime('%Y%m%d')}.txt"
            with open(stats_file, 'w', encoding='utf-8') as f:
                f.write("=== 향수 데이터셋 통계 ===\n\n")
                f.write(f"총 항목 수: {len(df)}\n")
                f.write(f"브랜드 수: {df['brand'].nunique()}\n")
                f.write(f"평균 평점: {df['rating'].mean():.2f}\n")
                f.write(f"가격 범위: {df['price'].min():.0f} - {df['price'].max():.0f}원\n")
                f.write(f"\n브랜드별 분포:\n")
                f.write(df['brand'].value_counts().head(10).to_string())
                f.write(f"\n\n성별 분포:\n")
                f.write(df['gender'].value_counts().to_string())

            logger.info(f"Dataset exported to {csv_file}")
            return str(csv_file)

        except Exception as e:
            logger.error(f"Dataset export failed: {e}")
            raise

async def main():
    """메인 실행 함수"""

    parser = argparse.ArgumentParser(description="Build fragrance dataset")
    parser.add_argument('--data-dir', default='./data', help='Data directory')
    parser.add_argument('--no-scraping', action='store_true', help='Disable web scraping')
    parser.add_argument('--sample-only', action='store_true', help='Use sample data only')
    parser.add_argument('--export-only', action='store_true', help='Only export existing dataset')

    args = parser.parse_args()

    builder = FragranceDatasetBuilder(
        data_dir=args.data_dir,
        enable_web_scraping=not args.no_scraping,
        use_sample_data=args.sample_only
    )

    try:
        if args.export_only:
            # 기존 데이터셋 내보내기만
            export_file = await builder.export_dataset_for_analysis()
            print(f"Dataset exported to: {export_file}")
        else:
            # 완전한 데이터셋 구축
            stats = await builder.build_complete_dataset()

            print("\n=== 데이터셋 구축 완료 ===")
            print(f"수집된 원본 데이터: {stats['raw_items_collected']}개")
            print(f"정제된 데이터: {stats['cleaned_items']}개")
            print(f"생성된 임베딩: {stats['embeddings_generated']}개")
            print(f"벡터 DB 항목: {stats['vector_db_items']}개")
            print(f"PostgreSQL 항목: {stats['database_items']}개")
            print(f"품질 점수: {stats['quality_score']:.2f}")
            print(f"총 소요 시간: {stats['total_duration']:.2f}초")

            # 분석용 내보내기
            export_file = await builder.export_dataset_for_analysis()
            print(f"분석용 데이터셋: {export_file}")

    except Exception as e:
        logger.error(f"Dataset build failed: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(asyncio.run(main()))