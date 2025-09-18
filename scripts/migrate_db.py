#!/usr/bin/env python3
"""
데이터베이스 마이그레이션 스크립트
SQLite에서 PostgreSQL로 데이터 이전
"""

import sqlite3
import asyncio
import asyncpg
import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatabaseMigrator:
    """데이터베이스 마이그레이션 클래스"""

    def __init__(
        self,
        sqlite_path: str = "./fragrance_ai.db",
        postgres_url: str = "postgresql://fragrance_ai:fragrance_ai_password@localhost:5432/fragrance_ai"
    ):
        self.sqlite_path = sqlite_path
        self.postgres_url = postgres_url
        self.migration_log = []

    async def migrate(self, backup: bool = True, dry_run: bool = False) -> Dict[str, Any]:
        """전체 마이그레이션 실행"""
        logger.info("🚀 Starting database migration...")

        # 백업 생성
        if backup and not dry_run:
            await self.create_backup()

        # SQLite 연결 확인
        if not os.path.exists(self.sqlite_path):
            raise FileNotFoundError(f"SQLite database not found: {self.sqlite_path}")

        # PostgreSQL 연결 확인
        pg_conn = None
        try:
            pg_conn = await asyncpg.connect(self.postgres_url)
            logger.info("✅ PostgreSQL connection established")
        except Exception as e:
            logger.error(f"❌ PostgreSQL connection failed: {e}")
            raise

        try:
            # 마이그레이션 실행
            migration_result = await self._perform_migration(pg_conn, dry_run)

            logger.info("✅ Migration completed successfully!")
            return migration_result

        except Exception as e:
            logger.error(f"❌ Migration failed: {e}")
            raise
        finally:
            if pg_conn:
                await pg_conn.close()

    async def _perform_migration(self, pg_conn: asyncpg.Connection, dry_run: bool = False) -> Dict[str, Any]:
        """실제 마이그레이션 수행"""
        sqlite_conn = sqlite3.connect(self.sqlite_path)
        sqlite_conn.row_factory = sqlite3.Row

        migration_stats = {
            "tables_migrated": 0,
            "records_migrated": 0,
            "errors": [],
            "start_time": datetime.now(),
            "dry_run": dry_run
        }

        try:
            # 테이블 목록 가져오기
            cursor = sqlite_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
            tables = [row[0] for row in cursor.fetchall()]

            logger.info(f"📋 Found {len(tables)} tables to migrate: {', '.join(tables)}")

            for table_name in tables:
                try:
                    # 테이블 스키마 정보 가져오기
                    schema_info = await self._get_table_schema(sqlite_conn, table_name)

                    if not dry_run:
                        # PostgreSQL 테이블 생성
                        await self._create_postgres_table(pg_conn, table_name, schema_info)

                    # 데이터 마이그레이션
                    records_count = await self._migrate_table_data(
                        sqlite_conn, pg_conn, table_name, dry_run
                    )

                    migration_stats["tables_migrated"] += 1
                    migration_stats["records_migrated"] += records_count

                    logger.info(f"✅ Migrated table '{table_name}': {records_count} records")

                except Exception as e:
                    error_msg = f"Failed to migrate table '{table_name}': {e}"
                    logger.error(f"❌ {error_msg}")
                    migration_stats["errors"].append(error_msg)
                    continue

            migration_stats["end_time"] = datetime.now()
            migration_stats["duration"] = (
                migration_stats["end_time"] - migration_stats["start_time"]
            ).total_seconds()

            return migration_stats

        finally:
            sqlite_conn.close()

    async def _get_table_schema(self, sqlite_conn: sqlite3.Connection, table_name: str) -> List[Dict[str, Any]]:
        """SQLite 테이블 스키마 정보 가져오기"""
        cursor = sqlite_conn.execute(f"PRAGMA table_info({table_name})")
        columns = []

        for row in cursor.fetchall():
            column_info = {
                "name": row[1],
                "type": self._convert_sqlite_type_to_postgres(row[2]),
                "nullable": not row[3],
                "default": row[4],
                "primary_key": bool(row[5])
            }
            columns.append(column_info)

        return columns

    def _convert_sqlite_type_to_postgres(self, sqlite_type: str) -> str:
        """SQLite 타입을 PostgreSQL 타입으로 변환"""
        type_mapping = {
            "INTEGER": "INTEGER",
            "TEXT": "TEXT",
            "REAL": "REAL",
            "BLOB": "BYTEA",
            "NUMERIC": "NUMERIC",
            "BOOLEAN": "BOOLEAN",
            "DATETIME": "TIMESTAMP",
            "DATE": "DATE",
            "JSON": "JSONB"
        }

        sqlite_type_upper = sqlite_type.upper()

        # VARCHAR(n) 처리
        if "VARCHAR" in sqlite_type_upper:
            return sqlite_type

        # 기본 타입 매핑
        for sqlite_t, postgres_t in type_mapping.items():
            if sqlite_t in sqlite_type_upper:
                return postgres_t

        # 기본값
        return "TEXT"

    async def _create_postgres_table(
        self,
        pg_conn: asyncpg.Connection,
        table_name: str,
        columns: List[Dict[str, Any]]
    ):
        """PostgreSQL 테이블 생성"""
        # 테이블이 이미 존재하는지 확인
        table_exists = await pg_conn.fetchval(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name = $1
            )
            """,
            table_name
        )

        if table_exists:
            logger.warning(f"⚠️  Table '{table_name}' already exists, skipping creation")
            return

        # 컬럼 정의 생성
        column_definitions = []
        for col in columns:
            col_def = f'"{col["name"]}" {col["type"]}'

            if not col["nullable"]:
                col_def += " NOT NULL"

            if col["default"]:
                col_def += f" DEFAULT {col['default']}"

            if col["primary_key"]:
                col_def += " PRIMARY KEY"

            column_definitions.append(col_def)

        # CREATE TABLE SQL 생성
        create_sql = f"""
        CREATE TABLE "{table_name}" (
            {', '.join(column_definitions)}
        )
        """

        await pg_conn.execute(create_sql)
        logger.info(f"📋 Created table '{table_name}'")

    async def _migrate_table_data(
        self,
        sqlite_conn: sqlite3.Connection,
        pg_conn: asyncpg.Connection,
        table_name: str,
        dry_run: bool = False
    ) -> int:
        """테이블 데이터 마이그레이션"""
        cursor = sqlite_conn.execute(f"SELECT * FROM {table_name}")
        rows = cursor.fetchall()

        if not rows:
            return 0

        if dry_run:
            logger.info(f"🔍 [DRY RUN] Would migrate {len(rows)} records from '{table_name}'")
            return len(rows)

        # 첫 번째 행으로부터 컬럼 이름 가져오기
        columns = [description[0] for description in cursor.description]

        # 배치 단위로 데이터 삽입
        batch_size = 1000
        total_inserted = 0

        for i in range(0, len(rows), batch_size):
            batch_rows = rows[i:i + batch_size]

            # INSERT SQL 생성
            placeholders = ', '.join([f'${j+1}' for j in range(len(columns))])
            insert_sql = f"""
            INSERT INTO "{table_name}" ({', '.join([f'"{col}"' for col in columns])})
            VALUES ({placeholders})
            """

            # 배치 삽입
            batch_data = []
            for row in batch_rows:
                # SQLite Row를 tuple로 변환
                row_data = tuple(row)
                batch_data.append(row_data)

            await pg_conn.executemany(insert_sql, batch_data)
            total_inserted += len(batch_data)

            logger.info(f"📦 Inserted batch: {total_inserted}/{len(rows)} records")

        return total_inserted

    async def create_backup(self):
        """현재 데이터베이스 백업 생성"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"./backups/fragrance_ai_backup_{timestamp}.db"

        # 백업 디렉토리 생성
        Path("./backups").mkdir(exist_ok=True)

        # SQLite 백업
        if os.path.exists(self.sqlite_path):
            import shutil
            shutil.copy2(self.sqlite_path, backup_path)
            logger.info(f"💾 Database backup created: {backup_path}")

        # 마이그레이션 로그 백업
        log_path = f"./backups/migration_log_{timestamp}.json"
        with open(log_path, 'w') as f:
            json.dump({
                "backup_time": timestamp,
                "original_db": self.sqlite_path,
                "backup_file": backup_path,
                "postgres_url": self.postgres_url.split('@')[1] if '@' in self.postgres_url else self.postgres_url
            }, f, indent=2)

        return backup_path

    async def verify_migration(self) -> Dict[str, Any]:
        """마이그레이션 검증"""
        logger.info("🔍 Verifying migration...")

        verification_result = {
            "sqlite_tables": {},
            "postgres_tables": {},
            "differences": []
        }

        # SQLite 테이블 정보
        sqlite_conn = sqlite3.connect(self.sqlite_path)
        cursor = sqlite_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        sqlite_tables = [row[0] for row in cursor.fetchall()]

        for table in sqlite_tables:
            count = sqlite_conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            verification_result["sqlite_tables"][table] = count

        sqlite_conn.close()

        # PostgreSQL 테이블 정보
        pg_conn = await asyncpg.connect(self.postgres_url)
        try:
            postgres_tables = await pg_conn.fetch(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
            )

            for table_row in postgres_tables:
                table_name = table_row['table_name']
                count = await pg_conn.fetchval(f'SELECT COUNT(*) FROM "{table_name}"')
                verification_result["postgres_tables"][table_name] = count
        finally:
            await pg_conn.close()

        # 차이점 분석
        for table, sqlite_count in verification_result["sqlite_tables"].items():
            postgres_count = verification_result["postgres_tables"].get(table, 0)
            if sqlite_count != postgres_count:
                verification_result["differences"].append({
                    "table": table,
                    "sqlite_count": sqlite_count,
                    "postgres_count": postgres_count,
                    "difference": sqlite_count - postgres_count
                })

        return verification_result


async def main():
    """메인 실행 함수"""
    import argparse

    parser = argparse.ArgumentParser(description='Database Migration Tool')
    parser.add_argument('--sqlite-path', default='./fragrance_ai.db', help='SQLite database path')
    parser.add_argument('--postgres-url',
                       default='postgresql://fragrance_ai:fragrance_ai_password@localhost:5432/fragrance_ai',
                       help='PostgreSQL connection URL')
    parser.add_argument('--no-backup', action='store_true', help='Skip backup creation')
    parser.add_argument('--dry-run', action='store_true', help='Dry run without actual migration')
    parser.add_argument('--verify-only', action='store_true', help='Only verify existing migration')

    args = parser.parse_args()

    migrator = DatabaseMigrator(args.sqlite_path, args.postgres_url)

    try:
        if args.verify_only:
            result = await migrator.verify_migration()
            print(json.dumps(result, indent=2))
        else:
            result = await migrator.migrate(
                backup=not args.no_backup,
                dry_run=args.dry_run
            )

            print("\n" + "="*50)
            print("📊 MIGRATION SUMMARY")
            print("="*50)
            print(f"Tables migrated: {result['tables_migrated']}")
            print(f"Records migrated: {result['records_migrated']}")
            print(f"Duration: {result['duration']:.2f} seconds")

            if result['errors']:
                print(f"\n❌ Errors: {len(result['errors'])}")
                for error in result['errors']:
                    print(f"  - {error}")
            else:
                print("\n✅ No errors")

    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))