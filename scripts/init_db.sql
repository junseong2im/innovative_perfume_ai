-- Fragrance AI 초기 데이터베이스 설정
-- PostgreSQL 초기화 스크립트

-- 확장 기능 활성화
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- 인덱스 성능 향상을 위한 설정
SET work_mem = '256MB';

-- 기본 역할 및 권한 설정 (이미 존재하는 경우 무시)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'fragrance_ai_readonly') THEN
        CREATE ROLE fragrance_ai_readonly;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'fragrance_ai_readwrite') THEN
        CREATE ROLE fragrance_ai_readwrite;
    END IF;
END
$$;

-- 기본 권한 부여
GRANT CONNECT ON DATABASE fragrance_ai TO fragrance_ai_readonly;
GRANT CONNECT ON DATABASE fragrance_ai TO fragrance_ai_readwrite;

-- 스키마 생성 (애플리케이션에서 생성되지만 보안을 위해 미리 생성)
CREATE SCHEMA IF NOT EXISTS public;
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS logs;

-- 권한 설정
GRANT USAGE ON SCHEMA public TO fragrance_ai_readonly, fragrance_ai_readwrite;
GRANT USAGE ON SCHEMA analytics TO fragrance_ai_readonly, fragrance_ai_readwrite;
GRANT USAGE ON SCHEMA logs TO fragrance_ai_readonly, fragrance_ai_readwrite;

-- 로깅 설정
ALTER SYSTEM SET log_statement = 'all';
ALTER SYSTEM SET log_min_duration_statement = 1000;  -- 1초 이상 쿼리 로깅

-- 성능 최적화 설정
ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements';
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET shared_buffers = '256MB';

-- 초기 데이터 삽입용 임시 테이블 (나중에 삭제됨)
CREATE TABLE IF NOT EXISTS temp_init_status (
    initialized_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    version VARCHAR(20) DEFAULT '1.0.0'
);

INSERT INTO temp_init_status (version) VALUES ('1.0.0')
ON CONFLICT DO NOTHING;

-- 초기 관리자 설정을 위한 임시 데이터
-- (실제 애플리케이션 시작 시 관리자 계정이 생성됨)
COMMENT ON DATABASE fragrance_ai IS 'Fragrance AI 시스템 데이터베이스 - v1.0.0';

-- 성능 모니터링을 위한 뷰 생성
CREATE OR REPLACE VIEW performance_stats AS
SELECT
    schemaname,
    tablename,
    attname,
    n_distinct,
    correlation
FROM pg_stats
WHERE schemaname = 'public';

-- 연결 상태 확인용 함수
CREATE OR REPLACE FUNCTION check_db_health()
RETURNS TEXT AS $$
BEGIN
    RETURN 'Database is healthy at ' || CURRENT_TIMESTAMP;
END;
$$ LANGUAGE plpgsql;

-- 초기화 완료 로그
DO $$
BEGIN
    RAISE NOTICE 'Fragrance AI Database initialized successfully at %', CURRENT_TIMESTAMP;
END
$$;