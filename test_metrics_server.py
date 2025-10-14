#!/usr/bin/env python3
"""
Fragrance AI 메트릭 테스트 서버
Grafana 대시보드 테스트용 샘플 메트릭 생성
"""

from prometheus_client import start_http_server, Gauge, Counter, Histogram
import random
import time

# RL 메트릭
rl_reward = Gauge('rl_reward', 'RL average reward', ['algorithm'])
rl_reward_ma = Gauge('rl_reward_ma', 'RL moving average reward', ['algorithm', 'window'])
rl_loss = Gauge('rl_loss', 'RL loss', ['loss_type'])
rl_entropy = Gauge('rl_entropy', 'RL policy entropy', ['algorithm'])
rl_clip_frac = Gauge('rl_clip_frac', 'RL clipping fraction', ['algorithm'])
rl_distill_kl = Gauge('rl_distill_kl', 'RL distillation KL divergence', ['teacher', 'student'])

# LLM 메트릭
llm_brief_total = Counter('llm_brief_total', 'LLM brief generation count', ['mode', 'status'])
llm_brief_repairs_total = Counter('llm_brief_repairs_total', 'LLM brief repair count', ['mode', 'repair_type'])
llm_brief_latency_seconds = Histogram('llm_brief_latency_seconds', 'LLM brief generation latency', ['mode'])
llm_brief_elapsed_ms = Histogram('llm_brief_elapsed_ms', 'LLM brief generation elapsed time (ms)', ['mode', 'model'])
llm_schema_fix_count_total = Counter('llm_schema_fix_count_total', 'LLM schema fix count', ['mode'])
hybrid_switch_total = Counter('hybrid_switch_total', 'Hybrid mode switch count', ['from_mode', 'to_mode'])
hybrid_switch_ratio = Gauge('hybrid_switch_ratio', 'Hybrid mode ratio', ['mode'])

# 규제 메트릭
ifra_violations_total = Counter('ifra_violations_total', 'IFRA regulation violations', ['ingredient', 'violation_type'])
allergen_hits_total = Counter('allergen_hits_total', 'Allergen detection hits', ['allergen', 'severity'])
ifra_compliance_rate = Gauge('ifra_compliance_rate', 'IFRA compliance rate', ['check_type'])

# Cloud Hub 메트릭
ipfs_store_rate = Gauge('ipfs_store_rate', 'IPFS storage rate (ops/sec)', ['operation'])
ipfs_restore_latency_seconds = Histogram('ipfs_restore_latency_seconds', 'IPFS restore latency', ['data_type'])
ipfs_cid_errors_total = Counter('ipfs_cid_errors_total', 'IPFS CID errors', ['error_type'])
redis_metadata_latency_seconds = Histogram('redis_metadata_latency_seconds', 'Redis metadata latency', ['operation'])

# API 메트릭
api_request_latency_seconds = Histogram('api_request_latency_seconds', 'API request latency', ['endpoint', 'method'])
api_requests_total = Counter('api_requests_total', 'Total API requests', ['endpoint', 'method', 'status'])

# 시스템 메트릭
system_error_rate = Gauge('system_error_rate', 'System error rate', ['component', 'error_type'])
system_rps = Gauge('system_rps', 'System requests per second', ['service'])
worker_vram_bytes = Gauge('worker_vram_bytes', 'Worker VRAM usage (bytes)', ['worker_id', 'model'])
worker_cpu_percent = Gauge('worker_cpu_percent', 'Worker CPU usage (%)', ['worker_id'])

# Cache 메트릭
cache_hit_rate = Gauge('cache_hit_rate', 'Cache hit rate', ['mode', 'cache_type'])

def generate_rl_metrics():
    """RL 메트릭 생성"""
    # PPO 알고리즘 메트릭
    rl_reward.labels(algorithm='ppo').set(random.uniform(15, 25))
    rl_reward_ma.labels(algorithm='ppo', window='100').set(random.uniform(18, 23))
    rl_reward_ma.labels(algorithm='ppo', window='1000').set(random.uniform(19, 22))
    rl_loss.labels(loss_type='policy_loss').set(random.uniform(0.1, 0.5))
    rl_loss.labels(loss_type='value_loss').set(random.uniform(0.2, 0.6))
    rl_loss.labels(loss_type='total_loss').set(random.uniform(0.3, 1.0))
    rl_entropy.labels(algorithm='ppo').set(random.uniform(1.5, 2.5))
    rl_clip_frac.labels(algorithm='ppo').set(random.uniform(0.05, 0.25))

    # REINFORCE 알고리즘 메트릭
    rl_reward.labels(algorithm='reinforce').set(random.uniform(12, 20))
    rl_reward_ma.labels(algorithm='reinforce', window='100').set(random.uniform(14, 18))

    # Distillation KL divergence
    rl_distill_kl.labels(teacher='ppo', student='llama_8b').set(random.uniform(0.10, 0.30))
    rl_distill_kl.labels(teacher='ppo', student='qwen_7b').set(random.uniform(0.12, 0.28))

def generate_llm_metrics():
    """LLM 메트릭 생성"""
    # Brief 생성 카운트
    modes = ['fast', 'balanced', 'creative']
    models = ['qwen', 'mistral', 'llama']

    for mode in modes:
        # 성공
        llm_brief_total.labels(mode=mode, status='success').inc(random.randint(1, 5))
        # 실패 (가끔)
        if random.random() < 0.1:
            llm_brief_total.labels(mode=mode, status='failure').inc(1)

    # Brief 수정 카운트
    repair_types = ['json_fix', 'schema_fix', 'validation_fix']
    for mode in modes:
        for repair_type in repair_types:
            if random.random() < 0.3:
                llm_brief_repairs_total.labels(mode=mode, repair_type=repair_type).inc(1)

    # Schema fix 카운트 (deployment gate용)
    for mode in modes:
        if random.random() < 0.05:  # 5% 확률로 스키마 에러
            llm_schema_fix_count_total.labels(mode=mode).inc(1)

    # Latency (seconds)
    llm_brief_latency_seconds.labels(mode='fast').observe(random.uniform(1.5, 2.3))
    llm_brief_latency_seconds.labels(mode='balanced').observe(random.uniform(2.0, 3.0))
    llm_brief_latency_seconds.labels(mode='creative').observe(random.uniform(3.0, 4.2))

    # Elapsed time (ms) - p95 계산용
    for mode in modes:
        for model in models:
            if mode == 'fast':
                llm_brief_elapsed_ms.labels(mode=mode, model=model).observe(random.uniform(800, 1200))
            elif mode == 'balanced':
                llm_brief_elapsed_ms.labels(mode=mode, model=model).observe(random.uniform(1500, 2500))
            else:  # creative
                llm_brief_elapsed_ms.labels(mode=mode, model=model).observe(random.uniform(2500, 4000))

    # Hybrid mode switch
    if random.random() < 0.1:  # 10% 확률로 모드 전환
        from_mode = random.choice(['exploration', 'exploitation'])
        to_mode = 'exploitation' if from_mode == 'exploration' else 'exploration'
        hybrid_switch_total.labels(from_mode=from_mode, to_mode=to_mode).inc(1)

    # Hybrid mode ratio
    hybrid_switch_ratio.labels(mode='exploration').set(random.uniform(0.25, 0.35))
    hybrid_switch_ratio.labels(mode='exploitation').set(random.uniform(0.65, 0.75))

def generate_api_metrics():
    """API 메트릭 생성"""
    endpoints = [
        ('/generate', 'POST'),
        ('/evolve', 'POST'),
        ('/health', 'GET'),
        ('/metrics', 'GET'),
    ]

    for endpoint, method in endpoints:
        # 성공 요청
        api_requests_total.labels(endpoint=endpoint, method=method, status='200').inc(random.randint(1, 10))

        # Latency
        if endpoint == '/generate':
            api_request_latency_seconds.labels(endpoint=endpoint, method=method).observe(random.uniform(1.0, 3.0))
        elif endpoint == '/evolve':
            api_request_latency_seconds.labels(endpoint=endpoint, method=method).observe(random.uniform(2.0, 5.0))
        else:
            api_request_latency_seconds.labels(endpoint=endpoint, method=method).observe(random.uniform(0.01, 0.1))

        # 실패 (가끔)
        if random.random() < 0.05:
            api_requests_total.labels(endpoint=endpoint, method=method, status='500').inc(1)

def generate_cache_metrics():
    """Cache 메트릭 생성"""
    cache_hit_rate.labels(mode='fast', cache_type='llm').set(random.uniform(0.60, 0.85))
    cache_hit_rate.labels(mode='balanced', cache_type='llm').set(random.uniform(0.55, 0.75))
    cache_hit_rate.labels(mode='creative', cache_type='llm').set(random.uniform(0.30, 0.50))

def generate_regulation_metrics():
    """규제 메트릭 생성"""
    # IFRA 위반 (매우 드물게)
    ingredients = ['coumarin', 'oakmoss', 'musk', 'bergamot']
    violation_types = ['concentration_exceeded', 'prohibited_combination', 'missing_declaration']

    for ingredient in ingredients:
        if random.random() < 0.01:  # 1% 확률로 위반
            violation_type = random.choice(violation_types)
            ifra_violations_total.labels(ingredient=ingredient, violation_type=violation_type).inc(1)

    # 알러젠 검출
    allergens = ['linalool', 'limonene', 'geraniol', 'citral']
    severities = ['low', 'medium', 'high']

    for allergen in allergens:
        if random.random() < 0.2:  # 20% 확률로 검출
            severity = random.choice(severities)
            allergen_hits_total.labels(allergen=allergen, severity=severity).inc(1)

    # IFRA 준수율
    ifra_compliance_rate.labels(check_type='ingredient').set(random.uniform(0.98, 1.0))
    ifra_compliance_rate.labels(check_type='concentration').set(random.uniform(0.95, 1.0))
    ifra_compliance_rate.labels(check_type='combination').set(random.uniform(0.97, 1.0))

def generate_hub_metrics():
    """Cloud Hub 메트릭 생성"""
    # IPFS storage rate
    ipfs_store_rate.labels(operation='store').set(random.uniform(5, 15))
    ipfs_store_rate.labels(operation='retrieve').set(random.uniform(10, 25))
    ipfs_store_rate.labels(operation='pin').set(random.uniform(2, 8))

    # IPFS restore latency
    data_types = ['feedback', 'checkpoint', 'recipe']
    for data_type in data_types:
        ipfs_restore_latency_seconds.labels(data_type=data_type).observe(random.uniform(0.1, 0.5))

    # IPFS CID errors (매우 드물게)
    error_types = ['invalid_cid', 'timeout', 'not_found']
    for error_type in error_types:
        if random.random() < 0.02:  # 2% 확률로 에러
            ipfs_cid_errors_total.labels(error_type=error_type).inc(1)

    # Redis metadata latency
    operations = ['hset', 'hget', 'expire']
    for operation in operations:
        redis_metadata_latency_seconds.labels(operation=operation).observe(random.uniform(0.001, 0.010))

def generate_system_metrics():
    """시스템 메트릭 생성"""
    # 에러율
    components = ['api', 'llm', 'worker', 'database']
    error_types = ['timeout', 'validation', 'internal', 'network']

    for component in components:
        for error_type in error_types:
            error_rate = random.uniform(0.0, 0.05)  # 0~5%
            system_error_rate.labels(component=component, error_type=error_type).set(error_rate)

    # RPS
    services = ['api', 'llm_ensemble', 'evolution', 'feedback']
    for service in services:
        rps = random.uniform(10, 100)
        system_rps.labels(service=service).set(rps)

    # 워커별 VRAM/CPU
    workers = ['worker_llm_1', 'worker_llm_2', 'worker_rl_1']
    models = ['qwen_32b', 'mistral_7b', 'llama_8b', 'ppo']

    for worker in workers:
        # VRAM (바이트)
        if 'llm' in worker:
            vram = random.uniform(12 * 1024**3, 18 * 1024**3)  # 12-18 GB
            model = random.choice(['qwen_32b', 'mistral_7b', 'llama_8b'])
        else:
            vram = random.uniform(4 * 1024**3, 8 * 1024**3)  # 4-8 GB
            model = 'ppo'

        worker_vram_bytes.labels(worker_id=worker, model=model).set(vram)

        # CPU (%)
        cpu = random.uniform(30, 85)
        worker_cpu_percent.labels(worker_id=worker).set(cpu)

def main():
    """메인 루프"""
    # Prometheus metrics server 시작 (포트 8000)
    print("Starting Prometheus metrics server on port 8000...")
    start_http_server(8000)
    print("[OK] Metrics server started at http://localhost:8000/metrics")
    print("[OK] Generating test metrics for Grafana dashboards...")

    while True:
        try:
            generate_rl_metrics()
            generate_llm_metrics()
            generate_api_metrics()
            generate_cache_metrics()
            generate_regulation_metrics()
            generate_hub_metrics()
            generate_system_metrics()

            # 10초마다 메트릭 업데이트
            time.sleep(10)

        except KeyboardInterrupt:
            print("\n[STOP] Shutting down metrics server...")
            break
        except Exception as e:
            print(f"[ERROR] Error: {e}")
            time.sleep(10)

if __name__ == '__main__':
    main()
