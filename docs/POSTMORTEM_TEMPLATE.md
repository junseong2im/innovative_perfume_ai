# Postmortem: [Incident Title]

**Incident ID:** [INC-YYYYMMDD-HHMMSS]
**Date:** [YYYY-MM-DD]
**Author(s):** [Name(s)]
**Severity:** [Sev1/Sev2/Sev3]
**Status:** [Draft/Review/Published]

---

## TL;DR (Executive Summary)

[2-3 문장으로 사건 요약]

**예시:**
> On October 14, 2025, the Qwen LLM model became unresponsive due to an out-of-memory condition, causing API latency to spike to 15 seconds (p95) and affecting approximately 25% of creative mode requests for 47 minutes. The system automatically downshifted to balanced mode, mitigating user impact while the root cause was identified and resolved.

---

## 1. Impact

### 사용자 영향
- **Affected Users:** [숫자 또는 비율]
- **Duration:** [시작 시간] ~ [종료 시간] ([총 시간])
- **Service Degradation:** [구체적 증상]
- **Revenue Impact:** [금액 또는 N/A]

**예시:**
- **Affected Users:** ~25% of creative mode users (~5,000 requests)
- **Duration:** 10:15 UTC ~ 11:02 UTC (47 minutes)
- **Service Degradation:** Creative mode API latency p95 increased from 3.8s to 15s
- **Revenue Impact:** Minimal (no billing affected)

### 기술적 영향
- **Systems Affected:** [영향받은 시스템/컴포넌트]
- **Data Loss:** [Yes/No - 구체적 설명]
- **SLO Impact:** [어떤 SLO 위반, 얼마나]

**예시:**
- **Systems Affected:** Qwen LLM service, Creative mode API
- **Data Loss:** No
- **SLO Impact:** LLM Creative p95 SLO violated (target: <4.5s, actual: 15s). Error budget consumed: 82% → 95%

---

## 2. Timeline (UTC)

| Time | Event | Action Taken |
|------|-------|--------------|
| 10:10 | [사건 시작] | - |
| 10:15 | [사건 감지] | [자동 알람/수동 발견] |
| 10:18 | [초기 대응] | [담당자 인지] |
| 10:25 | [조사] | [원인 조사 시작] |
| 10:40 | [완화] | [임시 해결] |
| 11:02 | [해결] | [근본 원인 해결] |
| 11:15 | [사건 종료] | [정상 확인] |

**예시:**
| Time | Event | Action Taken |
|------|-------|--------------|
| 10:10 | Qwen 메모리 사용량 급증 시작 (gradual leak) | - |
| 10:15 | Qwen health check 실패, 자동 알람 발생 | Prometheus alert triggered |
| 10:18 | On-call engineer (Alice) acknowledged | Started investigation |
| 10:22 | Qwen 서킷브레이커 활성화 | Automatic downshift: creative → balanced |
| 10:25 | OOM 근본 원인 확인 (memory leak in inference pipeline) | Identified root cause |
| 10:40 | Qwen 서비스 재시작 (increased memory limit) | Service restored |
| 10:55 | Creative mode 정상 작동 확인 | p95 latency: 3.9s |
| 11:02 | Incident resolved | All metrics normal |
| 11:15 | Postmortem initiated | Documentation started |

---

## 3. Root Cause

### 3.1 근본 원인 (What Happened)

[기술적으로 정확하게 무엇이 문제였는지 설명]

**예시:**
> A memory leak was introduced in the Qwen inference pipeline during the v2.1.3 deployment on October 12. The leak occurred in the attention mechanism caching layer, which failed to properly release memory after processing long sequences (> 2048 tokens). Over 48 hours, accumulated memory usage grew from 8GB to 15GB, eventually triggering an OOM condition and causing the Qwen process to become unresponsive.

### 3.2 왜 발생했는가 (Why It Happened)

[5 Whys 기법 또는 원인 분석]

**예시 (5 Whys):**
1. **Why did Qwen become unresponsive?**
   → OOM (Out of Memory) condition

2. **Why did OOM occur?**
   → Memory leak in attention mechanism caching

3. **Why was there a memory leak?**
   → Cache entries not properly released after long sequences

4. **Why weren't cache entries released?**
   → Incorrect reference counting in v2.1.3 refactoring

5. **Why wasn't this caught before production?**
   → Load tests used sequences < 1024 tokens; leak only manifest with longer sequences

### 3.3 왜 감지가 늦었는가 (Detection Delay)

[사건 발생부터 감지까지 지연 이유]

**예시:**
> The leak was gradual (48-hour accumulation) and memory usage was within normal operating range until the final 30 minutes. No alert threshold was configured for gradual memory growth patterns.

---

## 4. Resolution

### 4.1 즉각적 완화 조치 (Immediate Mitigation)

[사용자 영향 최소화를 위한 임시 조치]

**예시:**
1. **Automatic circuit breaker** activated at 10:22, routing creative → balanced
2. **Manual service restart** at 10:40 with increased memory limit (16GB → 24GB)
3. **Traffic validation** confirmed p95 latency restored to 3.9s

### 4.2 근본 원인 해결 (Root Cause Fix)

[장기적 해결책]

**예시:**
1. **Hotfix deployment** (v2.1.4) at 14:00 with corrected reference counting
2. **Memory monitoring** added for cache layer with alert threshold
3. **Load test enhancement** to include long sequences (up to 4096 tokens)

---

## 5. What Went Well

[사건 대응 중 잘된 점]

**예시:**
- ✅ **Automatic downshift** via circuit breaker limited user impact to 25%
- ✅ **Fast detection** (5 minutes from failure to alert)
- ✅ **Quick mitigation** (22 minutes to service restart)
- ✅ **Clear runbook** enabled on-call engineer to act decisively
- ✅ **No data loss** occurred

---

## 6. What Went Wrong

[사건 대응 중 문제점 - 비난이 아닌 시스템 개선 관점]

**예시:**
- ❌ **Gradual memory leak** not detected by existing monitors
- ❌ **Load tests insufficient** for long sequence scenarios
- ❌ **Memory limit too low** (16GB) given production traffic patterns
- ❌ **No automated memory profiling** in production
- ❌ **Communication delay** to user-facing status page (15 minutes)

---

## 7. Action Items

| Action | Owner | Priority | Due Date | Status |
|--------|-------|----------|----------|--------|
| [구체적 조치] | [담당자] | [P0/P1/P2] | [YYYY-MM-DD] | [Open/In Progress/Done] |

**예시:**
| Action | Owner | Priority | Due Date | Status |
|--------|-------|----------|----------|--------|
| Deploy hotfix v2.1.4 with corrected reference counting | Alice | P0 | 2025-10-14 | ✅ Done |
| Add memory growth rate alert (> 10% per hour) | Bob | P0 | 2025-10-15 | 🔄 In Progress |
| Increase Qwen memory limit to 24GB | Alice | P0 | 2025-10-14 | ✅ Done |
| Enhance load tests to include sequences up to 4096 tokens | Carol | P1 | 2025-10-18 | 📝 Open |
| Implement automated memory profiling in production | Dave | P1 | 2025-10-25 | 📝 Open |
| Add memory usage dashboard panel | Bob | P2 | 2025-10-20 | 📝 Open |
| Improve status page automation (auto-update on Sev2+) | Eve | P1 | 2025-10-22 | 📝 Open |
| Document memory leak debugging procedures in runbook | Alice | P2 | 2025-10-30 | 📝 Open |

**Priority Definitions:**
- **P0** - Critical, immediate action (within 24h)
- **P1** - High, prevent recurrence (within 1 week)
- **P2** - Medium, improve detection/response (within 1 month)

---

## 8. Lessons Learned

### 8.1 기술적 교훈

**예시:**
1. **Gradual degradation patterns** require different monitoring strategies than sudden failures
2. **Memory leaks** can be subtle and require long-running load tests to detect
3. **Circuit breakers** are highly effective at limiting blast radius
4. **Automatic downshift** preserved 75% of service availability during incident

### 8.2 프로세스 교훈

**예시:**
1. **Load test scenarios** should mirror production traffic patterns (including edge cases)
2. **Runbook automation** reduces response time significantly
3. **Status page updates** should be automated for Sev2+ incidents
4. **Memory monitoring** should include growth rate, not just absolute values

### 8.3 재발 방지

**예시:**
1. **Enhanced load testing** with long sequences and sustained traffic
2. **Memory profiling** in production environment (sampling mode)
3. **Proactive alerting** on memory growth trends
4. **Code review checklist** updated to include memory management verification

---

## 9. Supporting Information

### 9.1 관련 링크
- **Incident Ticket:** [Link to incident management system]
- **Grafana Dashboard:** [Link to relevant dashboard]
- **Logs:** [Link to log aggregation system]
- **Related PRs:** [Links to code changes]

**예시:**
- **Incident Ticket:** https://jira.company.com/INC-20251014-101500
- **Grafana Dashboard:** http://grafana/d/llm-health
- **Logs:** http://loki/explore?q={component="LLM",model="qwen"}
- **Hotfix PR:** https://github.com/company/artisan/pull/1234
- **Load Test PR:** https://github.com/company/artisan/pull/1235

### 9.2 메트릭 스냅샷

[사건 전후 주요 메트릭]

**예시:**
```
Before Incident (10:00):
  - Qwen memory: 8.2GB
  - Creative mode p95: 3.8s
  - Error rate: 0.12%

During Incident (10:30):
  - Qwen memory: 15.8GB (OOM)
  - Creative mode p95: 15.2s
  - Error rate: 8.5%

After Resolution (11:10):
  - Qwen memory: 8.5GB
  - Creative mode p95: 3.9s
  - Error rate: 0.15%
```

---

## 10. Sign-off

**Author:** [Name]
**Reviewed by:** [Name(s)]
**Approved by:** [Engineering Manager]
**Published:** [YYYY-MM-DD]

---

## Appendix

### A. Technical Details

[추가 기술 상세 정보, 스택 트레이스, 로그 등]

### B. Communication Timeline

[내부/외부 커뮤니케이션 타임라인]

---

**This postmortem follows the blameless postmortem principles:**
- ✅ No individual blame
- ✅ Focus on systems and processes
- ✅ Actionable improvements
- ✅ Learning culture
