.PHONY: help runbook-downshift runbook-rollback runbook-pin-update runbook-canary-promote runbook-emergency-stop

# Colors
YELLOW := \033[1;33m
GREEN := \033[1;32m
RED := \033[1;31m
NC := \033[0m

##@ Runbook Automation

help: ## Display this help message
	@echo "$(GREEN)=== Fragrance AI Runbook Automation ===$(NC)"
	@awk 'BEGIN {FS = ":.*##"; printf "\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  $(YELLOW)%-25s$(NC) %s\n", $$1, $$2 } /^##@/ { printf "\n$(GREEN)%s$(NC)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

runbook-downshift: ## Downshift: Reduce traffic to unstable service
	@echo "$(YELLOW)[Runbook] Initiating downshift...$(NC)"
	@python scripts/runbook_downshift.py --target fragrance_ai --reduce-to 50
	@echo "$(GREEN)[OK] Downshift complete. Traffic reduced to 50%.$(NC)"

runbook-rollback: ## Rollback: Revert to previous stable version
	@echo "$(YELLOW)[Runbook] Initiating rollback...$(NC)"
	@bash scripts/rollback.sh
	@echo "$(GREEN)[OK] Rollback complete.$(NC)"

runbook-pin-update: ## Pin update: Update model version pinning
	@echo "$(YELLOW)[Runbook] Updating model pins...$(NC)"
	@python scripts/runbook_pin_update.py --model qwen --version 2.5.1
	@python scripts/runbook_pin_update.py --model mistral --version 0.3.1
	@python scripts/runbook_pin_update.py --model llama --version 3.1.1
	@docker-compose restart fragrance_ai
	@echo "$(GREEN)[OK] Model pins updated and service restarted.$(NC)"

runbook-canary-promote: ## Canary promote: Promote canary to production
	@echo "$(YELLOW)[Runbook] Promoting canary to production...$(NC)"
	@bash scripts/canary_deployment.py --promote
	@bash scripts/update_nginx_weights.sh --canary 0 --stable 100
	@echo "$(GREEN)[OK] Canary promoted to production.$(NC)"

runbook-emergency-stop: ## Emergency stop: Stop all services immediately
	@echo "$(RED)[EMERGENCY] Stopping all services...$(NC)"
	@docker-compose down
	@echo "$(RED)[OK] All services stopped.$(NC)"

runbook-restart-workers: ## Restart workers: Restart LLM and RL workers
	@echo "$(YELLOW)[Runbook] Restarting workers...$(NC)"
	@docker-compose restart celery_worker celery_beat
	@echo "$(GREEN)[OK] Workers restarted.$(NC)"

runbook-clear-cache: ## Clear cache: Clear Redis cache
	@echo "$(YELLOW)[Runbook] Clearing cache...$(NC)"
	@docker-compose exec redis redis-cli FLUSHALL
	@echo "$(GREEN)[OK] Cache cleared.$(NC)"

runbook-health-check: ## Health check: Check all services health
	@echo "$(YELLOW)[Runbook] Running health checks...$(NC)"
	@curl -s http://localhost:8000/health | jq .
	@curl -s http://localhost:8002/health/llm?model=qwen | jq .
	@curl -s http://localhost:8002/health/llm?model=mistral | jq .
	@curl -s http://localhost:8002/health/llm?model=llama | jq .
	@echo "$(GREEN)[OK] Health checks complete.$(NC)"

runbook-metrics-check: ## Metrics check: Check Prometheus metrics
	@echo "$(YELLOW)[Runbook] Checking metrics...$(NC)"
	@curl -s http://localhost:9090/api/v1/query?query=up | jq .
	@echo "$(GREEN)[OK] Metrics check complete.$(NC)"

runbook-shadow-eval: ## Shadow evaluation: Run shadow evaluation
	@echo "$(YELLOW)[Runbook] Running shadow evaluation...$(NC)"
	@python fragrance_ai/deployment/shadow_evaluation.py
	@echo "$(GREEN)[OK] Shadow evaluation complete. Results in shadow_evaluation_results.json$(NC)"

##@ Development

dev-start: ## Start development environment
	@echo "$(GREEN)[DEV] Starting development environment...$(NC)"
	@docker-compose up -d
	@python test_metrics_server.py &
	@python health_api_server.py &
	@echo "$(GREEN)[OK] Development environment started.$(NC)"

dev-stop: ## Stop development environment
	@echo "$(YELLOW)[DEV] Stopping development environment...$(NC)"
	@docker-compose down
	@pkill -f test_metrics_server.py || true
	@pkill -f health_api_server.py || true
	@echo "$(GREEN)[OK] Development environment stopped.$(NC)"

dev-logs: ## Show development logs
	@docker-compose logs -f --tail=100

##@ Testing

test-all: ## Run all tests
	@echo "$(YELLOW)[TEST] Running all tests...$(NC)"
	@pytest tests/ -v
	@echo "$(GREEN)[OK] All tests passed.$(NC)"

test-okr: ## Run OKR validation tests
	@echo "$(YELLOW)[TEST] Running OKR validation tests...$(NC)"
	@pytest tests/test_advanced_ai_okr.py -v
	@echo "$(GREEN)[OK] OKR tests passed.$(NC)"

test-smoke: ## Run smoke tests
	@echo "$(YELLOW)[TEST] Running smoke tests...$(NC)"
	@bash scripts/smoke_test_manual.sh
	@echo "$(GREEN)[OK] Smoke tests passed.$(NC)"

##@ Monitoring

monitor-grafana: ## Open Grafana dashboard
	@echo "$(GREEN)[MON] Opening Grafana at http://localhost:3000$(NC)"
	@start http://localhost:3000

monitor-prometheus: ## Open Prometheus UI
	@echo "$(GREEN)[MON] Opening Prometheus at http://localhost:9090$(NC)"
	@start http://localhost:9090

monitor-metrics: ## Show current metrics
	@curl -s http://localhost:8000/metrics | grep -E "rl_reward|llm_brief|ifra_violations|ipfs_store"

##@ Deployment

deploy-staging: ## Deploy to staging
	@echo "$(YELLOW)[DEPLOY] Deploying to staging...$(NC)"
	@docker-compose -f docker-compose.yml up -d --build
	@echo "$(GREEN)[OK] Deployed to staging.$(NC)"

deploy-production: ## Deploy to production (with gate check)
	@echo "$(YELLOW)[DEPLOY] Running deployment gate check...$(NC)"
	@python scripts/pre_deployment_check.py
	@echo "$(YELLOW)[DEPLOY] Deploying to production...$(NC)"
	@bash scripts/deploy_blue_green.sh
	@echo "$(GREEN)[OK] Deployed to production.$(NC)"

deploy-canary: ## Deploy canary version
	@echo "$(YELLOW)[DEPLOY] Deploying canary (10% traffic)...$(NC)"
	@bash scripts/deploy_canary.sh
	@echo "$(GREEN)[OK] Canary deployed.$(NC)"

##@ Maintenance

backup-db: ## Backup database
	@echo "$(YELLOW)[MAINT] Backing up database...$(NC)"
	@docker-compose exec postgres pg_dump -U postgres fragrance_ai > backup_$$(date +%Y%m%d_%H%M%S).sql
	@echo "$(GREEN)[OK] Database backed up.$(NC)"

restore-db: ## Restore database from backup
	@echo "$(RED)[MAINT] Restoring database from backup...$(NC)"
	@read -p "Enter backup file: " backup_file; \
	docker-compose exec -T postgres psql -U postgres fragrance_ai < $$backup_file
	@echo "$(GREEN)[OK] Database restored.$(NC)"

clean-docker: ## Clean unused Docker resources
	@echo "$(YELLOW)[MAINT] Cleaning Docker resources...$(NC)"
	@docker system prune -f
	@echo "$(GREEN)[OK] Docker cleaned.$(NC)"
