import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import subprocess
import json
from dataclasses import dataclass
from enum import Enum

from alembic.config import Config
from alembic import command
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, text, inspect

from ..core.config import settings
from ..core.logging_config import get_logger, performance_logger
from ..core.exceptions import DatabaseException, SystemException, ErrorCode
from .connection import db_connection_manager

logger = get_logger(__name__)


class MigrationStatus(str, Enum):
    """마이그레이션 상태"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class MigrationInfo:
    """마이그레이션 정보"""
    revision_id: str
    description: str
    version: Optional[str]
    status: MigrationStatus
    applied_at: Optional[datetime]
    execution_time_ms: Optional[float]
    error_message: Optional[str] = None


class DatabaseMigrationManager:
    """데이터베이스 마이그레이션 관리자"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.alembic_config_path = self.project_root / "alembic.ini"
        self.migrations_dir = self.project_root / "migrations"
        
        self.alembic_cfg = None
        self._initialize_alembic()
    
    def _initialize_alembic(self):
        """Alembic 초기화"""
        
        try:
            if not self.alembic_config_path.exists():
                logger.warning("Alembic config not found, creating default configuration")
                self._create_alembic_config()
            
            self.alembic_cfg = Config(str(self.alembic_config_path))
            self.alembic_cfg.set_main_option("sqlalchemy.url", settings.database_url)
            
            # 스크립트 위치 설정
            self.alembic_cfg.set_main_option("script_location", str(self.migrations_dir))
            
            logger.info("Alembic configuration initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Alembic: {e}")
            raise DatabaseException(
                message=f"Alembic initialization failed: {str(e)}",
                cause=e
            )
    
    def _create_alembic_config(self):
        """Alembic 설정 파일 생성"""
        
        alembic_ini_content = f"""# A generic, single database configuration.

[alembic]
# path to migration scripts
script_location = migrations

# template used to generate migration file names; The default value is %%(rev)s_%%(slug)s
# Uncomment the line below if you want the files to be prepended with date and time
file_template = %%(year)d%%(month).2d%%(day).2d_%%(hour).2d%%(minute).2d_%%(rev)s_%%(slug)s

# sys.path path, will be prepended to sys.path if present.
# defaults to the current working directory.
prepend_sys_path = .

# timezone to use when rendering the date within the migration file
# as well as the filename.
# If specified, requires the python-dateutil library that can be
# installed by adding `alembic[tz]` to the pip requirements
# string value is passed to dateutil.tz.gettz()
# leave blank for localtime
# timezone =

# max length of characters to apply to the
# "slug" field
# truncate_slug_length = 40

# set to 'true' to run the environment during
# the 'revision' command, regardless of autogenerate
# revision_environment = false

# set to 'true' to allow .pyc and .pyo files without
# a source .py file to be detected as revisions in the
# versions/ directory
# sourceless = false

# version number format.  This value may contain strftime
# formatting directives.
version_num_format = %%(year)d%%(month).2d%%(day).2d_%%(hour).2d%%(minute).2d

# version path separator; As mentioned above, this is the character used to split
# version_locations. The default within new alembic.ini files is "os", which uses
# os.pathsep. If this key is omitted entirely, it falls back to the legacy
# behavior of splitting on spaces and/or commas.
# Valid values for version_path_separator are:
#
# version_path_separator = :
# version_path_separator = ;
# version_path_separator = space
version_path_separator = os

# set to 'true' to search source files recursively
# in each "version_locations" directory
# new in Alembic version 1.10
# recursive_version_locations = false

# the output encoding used when revision files
# are written from script.py.mako
# output_encoding = utf-8

sqlalchemy.url = {settings.database_url}

[post_write_hooks]
# post_write_hooks defines scripts or Python functions that are run
# on newly generated revision scripts.  See the documentation for further
# detail and examples

# format using "black" - use the console_scripts runner, against the "black" entrypoint
# hooks = black
# black.type = console_scripts
# black.entrypoint = black
# black.options = -l 79 REVISION_SCRIPT_FILENAME

# lint with attempts to fix using "ruff" - use the exec runner, execute a binary
# hooks = ruff
# ruff.type = exec
# ruff.executable = %(here)s/.venv/bin/ruff
# ruff.options = --fix REVISION_SCRIPT_FILENAME

# Logging configuration
[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
"""
        
        try:
            with open(self.alembic_config_path, 'w', encoding='utf-8') as f:
                f.write(alembic_ini_content)
            
            logger.info(f"Created Alembic configuration at {self.alembic_config_path}")
            
        except Exception as e:
            logger.error(f"Failed to create Alembic config: {e}")
            raise
    
    def initialize_migrations(self, force: bool = False) -> bool:
        """마이그레이션 환경 초기화"""
        
        try:
            if self.migrations_dir.exists() and not force:
                logger.info("Migrations directory already exists")
                return False
            
            if force and self.migrations_dir.exists():
                import shutil
                shutil.rmtree(self.migrations_dir)
            
            # Alembic init
            command.init(self.alembic_cfg, str(self.migrations_dir))
            
            # env.py 파일 수정 (프로젝트에 맞게)
            self._update_env_py()
            
            logger.info("Migration environment initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize migrations: {e}")
            raise DatabaseException(
                message=f"Migration initialization failed: {str(e)}",
                cause=e
            )
    
    def _update_env_py(self):
        """env.py 파일 업데이트"""
        
        env_py_path = self.migrations_dir / "env.py"
        
        if not env_py_path.exists():
            logger.warning("env.py not found, skipping update")
            return
        
        # env.py 내용을 프로젝트에 맞게 수정
        env_py_content = f'''from logging.config import fileConfig
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context
import sys
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 모델 import
from fragrance_ai.database.models import Base
from fragrance_ai.core.config import settings

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata

def get_url():
    return settings.database_url

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={{"paramstyle": "named"}},
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = get_url()
    
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
'''
        
        try:
            with open(env_py_path, 'w', encoding='utf-8') as f:
                f.write(env_py_content)
                
            logger.info("Updated env.py file")
            
        except Exception as e:
            logger.error(f"Failed to update env.py: {e}")
    
    def create_migration(
        self, 
        message: str, 
        auto_generate: bool = True,
        sql_mode: bool = False
    ) -> str:
        """새 마이그레이션 생성"""
        
        try:
            logger.info(f"Creating migration: {message}")
            
            start_time = datetime.now()
            
            if auto_generate:
                # 자동 생성 모드
                revision = command.revision(
                    self.alembic_cfg,
                    message=message,
                    autogenerate=True,
                    sql=sql_mode
                )
            else:
                # 수동 생성 모드
                revision = command.revision(
                    self.alembic_cfg,
                    message=message,
                    sql=sql_mode
                )
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            logger.info(f"Migration created successfully: {revision.revision}")
            
            performance_logger.log_execution_time(
                operation="create_migration",
                execution_time=execution_time,
                success=True,
                extra_data={
                    "revision_id": revision.revision,
                    "message": message,
                    "auto_generate": auto_generate
                }
            )
            
            return revision.revision
            
        except Exception as e:
            logger.error(f"Failed to create migration: {e}")
            raise DatabaseException(
                message=f"Migration creation failed: {str(e)}",
                cause=e
            )
    
    def apply_migrations(
        self, 
        target_revision: Optional[str] = None,
        dry_run: bool = False
    ) -> List[MigrationInfo]:
        """마이그레이션 적용"""
        
        try:
            logger.info(f"Applying migrations to revision: {target_revision or 'head'}")
            
            start_time = datetime.now()
            applied_migrations = []
            
            # 현재 상태 확인
            current_revision = self.get_current_revision()
            pending_migrations = self.get_pending_migrations()
            
            if not pending_migrations:
                logger.info("No pending migrations found")
                return []
            
            # 드라이런 모드
            if dry_run:
                logger.info("Dry run mode - showing SQL that would be executed")
                command.upgrade(
                    self.alembic_cfg,
                    target_revision or "head",
                    sql=True
                )
                return pending_migrations
            
            # 실제 마이그레이션 실행
            for migration in pending_migrations:
                migration_start = datetime.now()
                
                try:
                    # 개별 마이그레이션 적용
                    command.upgrade(self.alembic_cfg, migration.revision_id)
                    
                    migration_time = (datetime.now() - migration_start).total_seconds() * 1000
                    
                    migration.status = MigrationStatus.COMPLETED
                    migration.applied_at = datetime.now()
                    migration.execution_time_ms = migration_time
                    
                    applied_migrations.append(migration)
                    
                    logger.info(f"Migration {migration.revision_id} applied successfully")
                    
                    # 타겟 리비전에 도달했으면 중단
                    if target_revision and migration.revision_id == target_revision:
                        break
                        
                except Exception as migration_error:
                    migration.status = MigrationStatus.FAILED
                    migration.error_message = str(migration_error)
                    applied_migrations.append(migration)
                    
                    logger.error(f"Migration {migration.revision_id} failed: {migration_error}")
                    
                    # 실패한 마이그레이션이 있으면 중단
                    raise DatabaseException(
                        message=f"Migration {migration.revision_id} failed: {str(migration_error)}",
                        cause=migration_error
                    )
            
            total_time = (datetime.now() - start_time).total_seconds() * 1000
            
            logger.info(f"Applied {len(applied_migrations)} migrations successfully")
            
            performance_logger.log_execution_time(
                operation="apply_migrations",
                execution_time=total_time,
                success=True,
                extra_data={
                    "applied_count": len(applied_migrations),
                    "target_revision": target_revision
                }
            )
            
            return applied_migrations
            
        except Exception as e:
            logger.error(f"Failed to apply migrations: {e}")
            raise DatabaseException(
                message=f"Migration application failed: {str(e)}",
                cause=e
            )
    
    def rollback_migration(
        self,
        target_revision: str,
        dry_run: bool = False
    ) -> List[MigrationInfo]:
        """마이그레이션 롤백"""
        
        try:
            logger.info(f"Rolling back to revision: {target_revision}")
            
            start_time = datetime.now()
            
            if dry_run:
                logger.info("Dry run mode - showing SQL that would be executed")
                command.downgrade(self.alembic_cfg, target_revision, sql=True)
                return []
            
            # 실제 롤백 실행
            command.downgrade(self.alembic_cfg, target_revision)
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            logger.info(f"Rollback completed successfully")
            
            performance_logger.log_execution_time(
                operation="rollback_migration",
                execution_time=execution_time,
                success=True,
                extra_data={"target_revision": target_revision}
            )
            
            # 롤백된 마이그레이션 정보 반환
            return [MigrationInfo(
                revision_id=target_revision,
                description=f"Rollback to {target_revision}",
                version=None,
                status=MigrationStatus.ROLLED_BACK,
                applied_at=datetime.now(),
                execution_time_ms=execution_time
            )]
            
        except Exception as e:
            logger.error(f"Failed to rollback migrations: {e}")
            raise DatabaseException(
                message=f"Migration rollback failed: {str(e)}",
                cause=e
            )
    
    def get_current_revision(self) -> Optional[str]:
        """현재 리비전 조회"""
        
        try:
            engine = db_connection_manager.engine
            if not engine:
                raise DatabaseException("Database engine not initialized")
            
            with engine.connect() as conn:
                context = MigrationContext.configure(conn)
                return context.get_current_revision()
                
        except Exception as e:
            logger.error(f"Failed to get current revision: {e}")
            return None
    
    def get_migration_history(self) -> List[MigrationInfo]:
        """마이그레이션 히스토리 조회"""
        
        try:
            script = ScriptDirectory.from_config(self.alembic_cfg)
            
            # 모든 리비전 조회
            revisions = []
            for rev in script.walk_revisions():
                migration_info = MigrationInfo(
                    revision_id=rev.revision,
                    description=rev.doc or "No description",
                    version=getattr(rev, 'version', None),
                    status=MigrationStatus.PENDING,
                    applied_at=None,
                    execution_time_ms=None
                )
                revisions.append(migration_info)
            
            # 적용된 리비전 확인
            current_revision = self.get_current_revision()
            if current_revision:
                for migration in revisions:
                    if migration.revision_id == current_revision:
                        migration.status = MigrationStatus.COMPLETED
                        break
            
            return list(reversed(revisions))  # 최신 순으로 정렬
            
        except Exception as e:
            logger.error(f"Failed to get migration history: {e}")
            return []
    
    def get_pending_migrations(self) -> List[MigrationInfo]:
        """대기 중인 마이그레이션 조회"""
        
        try:
            script = ScriptDirectory.from_config(self.alembic_cfg)
            current_revision = self.get_current_revision()
            
            pending = []
            for rev in script.walk_revisions(current_revision, "head"):
                if rev.revision != current_revision:
                    migration_info = MigrationInfo(
                        revision_id=rev.revision,
                        description=rev.doc or "No description",
                        version=getattr(rev, 'version', None),
                        status=MigrationStatus.PENDING,
                        applied_at=None,
                        execution_time_ms=None
                    )
                    pending.append(migration_info)
            
            return list(reversed(pending))  # 적용 순서로 정렬
            
        except Exception as e:
            logger.error(f"Failed to get pending migrations: {e}")
            return []
    
    def validate_database_schema(self) -> Dict[str, Any]:
        """데이터베이스 스키마 검증"""
        
        try:
            logger.info("Validating database schema...")
            
            engine = db_connection_manager.engine
            if not engine:
                raise DatabaseException("Database engine not initialized")
            
            # 스키마 검사
            inspector = inspect(engine)
            
            # 테이블 목록
            tables = inspector.get_table_names()
            
            # 인덱스 정보
            indexes = {}
            for table in tables:
                indexes[table] = inspector.get_indexes(table)
            
            # 외래키 정보
            foreign_keys = {}
            for table in tables:
                foreign_keys[table] = inspector.get_foreign_keys(table)
            
            # 현재 리비전과 비교
            current_revision = self.get_current_revision()
            pending_migrations = self.get_pending_migrations()
            
            validation_result = {
                "status": "valid" if not pending_migrations else "pending_migrations",
                "current_revision": current_revision,
                "pending_migrations_count": len(pending_migrations),
                "tables_count": len(tables),
                "tables": tables,
                "indexes": indexes,
                "foreign_keys": foreign_keys,
                "validation_time": datetime.now().isoformat()
            }
            
            logger.info(f"Schema validation completed - Status: {validation_result['status']}")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Schema validation failed: {e}")
            raise DatabaseException(
                message=f"Schema validation failed: {str(e)}",
                cause=e
            )
    
    def backup_database(self, backup_path: Optional[str] = None) -> str:
        """데이터베이스 백업"""
        
        try:
            if backup_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_filename = f"fragrance_ai_backup_{timestamp}.sql"
                backup_path = str(Path("backups") / backup_filename)
            
            # 백업 디렉토리 생성
            Path(backup_path).parent.mkdir(exist_ok=True)
            
            logger.info(f"Creating database backup: {backup_path}")
            
            # PostgreSQL인 경우 pg_dump 사용
            if settings.database_url.startswith("postgresql"):
                self._postgresql_backup(backup_path)
            else:
                # SQLite의 경우 파일 복사
                self._sqlite_backup(backup_path)
            
            logger.info(f"Database backup completed: {backup_path}")
            
            return backup_path
            
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            raise DatabaseException(
                message=f"Database backup failed: {str(e)}",
                cause=e
            )
    
    def _postgresql_backup(self, backup_path: str):
        """PostgreSQL 백업"""
        
        import urllib.parse
        
        parsed = urllib.parse.urlparse(settings.database_url)
        
        env = os.environ.copy()
        env['PGPASSWORD'] = parsed.password
        
        cmd = [
            'pg_dump',
            '-h', parsed.hostname,
            '-p', str(parsed.port or 5432),
            '-U', parsed.username,
            '-d', parsed.path[1:],  # '/' 제거
            '-f', backup_path,
            '--no-password'
        ]
        
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise DatabaseException(f"pg_dump failed: {result.stderr}")
    
    def _sqlite_backup(self, backup_path: str):
        """SQLite 백업"""
        
        import shutil
        import urllib.parse
        
        parsed = urllib.parse.urlparse(settings.database_url)
        db_path = parsed.path
        
        if db_path.startswith('/'):
            db_path = db_path[1:]  # '/' 제거
        
        shutil.copy2(db_path, backup_path)


# 전역 마이그레이션 관리자 인스턴스
migration_manager = DatabaseMigrationManager()