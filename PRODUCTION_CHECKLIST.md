# ✅ Production Deployment Checklist

## 🔐 Security
- [ ] Generate new SECRET_KEY (minimum 64 characters)
- [ ] Generate new JWT_SECRET_KEY
- [ ] Change all default passwords
- [ ] Configure SSL certificates
- [ ] Set up firewall rules (only open necessary ports)
- [ ] Enable rate limiting
- [ ] Configure CORS for your domain only
- [ ] Disable DEBUG mode
- [ ] Remove or secure all test endpoints
- [ ] Enable audit logging

## 🗄️ Database
- [ ] Change default PostgreSQL password
- [ ] Create production database user with limited privileges
- [ ] Enable pgvector extension
- [ ] Run all migrations
- [ ] Create database backups schedule
- [ ] Test restore procedure
- [ ] Set up connection pooling
- [ ] Configure appropriate indexes

## 🔧 Configuration
- [ ] Copy .env.production to .env
- [ ] Update DATABASE_URL with production credentials
- [ ] Configure REDIS_URL
- [ ] Set OLLAMA_BASE_URL
- [ ] Update DOMAIN_NAME
- [ ] Configure email settings (if needed)
- [ ] Set appropriate LOG_LEVEL (info or warning)
- [ ] Configure feature flags

## 🐳 Docker
- [ ] Build all Docker images
- [ ] Test docker-compose.production.yml locally
- [ ] Push images to registry
- [ ] Configure resource limits
- [ ] Set up health checks
- [ ] Configure restart policies
- [ ] Test rolling updates

## 📊 Monitoring
- [ ] Set up Prometheus
- [ ] Configure Grafana dashboards
- [ ] Set up alerting rules
- [ ] Configure log aggregation
- [ ] Set up uptime monitoring
- [ ] Configure error tracking (Sentry, etc.)
- [ ] Set up performance monitoring

## 🧪 Testing
- [ ] Run all unit tests
- [ ] Run integration tests
- [ ] Perform load testing
- [ ] Test all API endpoints
- [ ] Verify health check endpoints
- [ ] Test database connections
- [ ] Test Redis connectivity
- [ ] Verify LLM services

## 🚀 Deployment
- [ ] Create deployment user
- [ ] Set up SSH keys
- [ ] Configure GitHub Actions secrets
- [ ] Test CI/CD pipeline
- [ ] Perform test deployment
- [ ] Verify all services start correctly
- [ ] Check application logs
- [ ] Test rollback procedure

## 📝 Documentation
- [ ] Update API documentation
- [ ] Document environment variables
- [ ] Create runbook for common issues
- [ ] Document backup/restore procedures
- [ ] Update README with production info
- [ ] Document scaling procedures
- [ ] Create incident response plan

## 🔄 Post-Deployment
- [ ] Verify all health checks pass
- [ ] Test critical user flows
- [ ] Monitor resource usage
- [ ] Check error rates
- [ ] Verify logging works
- [ ] Test alerting
- [ ] Perform security scan
- [ ] Schedule first backup

## 📱 Notifications
- [ ] Set up deployment notifications
- [ ] Configure error alerts
- [ ] Set up performance alerts
- [ ] Configure backup notifications
- [ ] Test all notification channels

## 🎯 Performance
- [ ] Enable Redis caching
- [ ] Configure CDN (if applicable)
- [ ] Optimize database queries
- [ ] Enable compression
- [ ] Configure appropriate timeouts
- [ ] Set up rate limiting
- [ ] Test under load

## 🔒 Compliance
- [ ] GDPR compliance (if applicable)
- [ ] Data retention policies
- [ ] Privacy policy updated
- [ ] Terms of service updated
- [ ] Security headers configured
- [ ] HTTPS enforced

---

## Final Verification Commands

```bash
# Health check
curl https://yourdomain.com/health/detailed

# Check all services
docker-compose -f docker-compose.production.yml ps

# View logs
docker-compose -f docker-compose.production.yml logs --tail=100

# Database connection
docker-compose -f docker-compose.production.yml exec postgres psql -U fragrance -c "SELECT version();"

# Redis connection
docker-compose -f docker-compose.production.yml exec redis redis-cli ping

# API documentation
curl https://yourdomain.com/docs
```

## Emergency Contacts

- DevOps Lead: ___________
- Database Admin: ___________
- Security Team: ___________
- On-call Engineer: ___________

---

**Sign-off**

- [ ] Development Team Lead
- [ ] DevOps Engineer
- [ ] Security Officer
- [ ] Product Owner

**Deployment Date**: ___________
**Deployed By**: ___________
**Version**: ___________