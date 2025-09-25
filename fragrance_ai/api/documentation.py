"""
Comprehensive API Documentation System
Auto-generated interactive documentation with examples
"""

from typing import Dict, List, Any, Optional, Union
from fastapi import FastAPI, Request
from fastapi.openapi.utils import get_openapi
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.responses import HTMLResponse
import json
import os
from datetime import datetime
from pathlib import Path

from ..core.config import settings
from ..core.production_logging import get_logger

logger = get_logger(__name__)


class APIDocumentationGenerator:
    """Advanced API documentation generator"""

    def __init__(self, app: FastAPI):
        self.app = app
        self.custom_openapi_schema = None

    def generate_enhanced_openapi_schema(self) -> Dict[str, Any]:
        """Generate enhanced OpenAPI schema with comprehensive details"""

        if self.custom_openapi_schema:
            return self.custom_openapi_schema

        openapi_schema = get_openapi(
            title="Fragrance AI API",
            version="3.0.0",
            description=self._get_api_description(),
            routes=self.app.routes,
            servers=[
                {
                    "url": "https://api.fragranceai.com",
                    "description": "Production server"
                },
                {
                    "url": "https://staging-api.fragranceai.com",
                    "description": "Staging server"
                },
                {
                    "url": os.environ.get("API_BASE_URL", "http://localhost:8000"),
                    "description": "Development server"
                }
            ]
        )

        # Add comprehensive metadata
        openapi_schema["info"].update({
            "contact": {
                "name": "Fragrance AI Support",
                "url": "https://fragranceai.com/support",
                "email": "support@fragranceai.com"
            },
            "license": {
                "name": "Proprietary License",
                "url": "https://fragranceai.com/license"
            },
            "termsOfService": "https://fragranceai.com/terms",
            "x-logo": {
                "url": "https://fragranceai.com/logo.png",
                "altText": "Fragrance AI Logo"
            }
        })

        # Add security schemes
        openapi_schema["components"]["securitySchemes"] = {
            "ApiKeyAuth": {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key",
                "description": "API key for authentication"
            },
            "BearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT",
                "description": "JWT token authentication"
            }
        }

        # Add global security requirements
        openapi_schema["security"] = [
            {"ApiKeyAuth": []},
            {"BearerAuth": []}
        ]

        # Add comprehensive tags
        openapi_schema["tags"] = [
            {
                "name": "Search",
                "description": "Semantic fragrance search operations",
                "externalDocs": {
                    "description": "Search Guide",
                    "url": "https://docs.fragranceai.com/search"
                }
            },
            {
                "name": "Generation",
                "description": "AI-powered fragrance recipe generation",
                "externalDocs": {
                    "description": "Generation Guide",
                    "url": "https://docs.fragranceai.com/generation"
                }
            },
            {
                "name": "Training",
                "description": "Model training and fine-tuning operations",
                "externalDocs": {
                    "description": "Training Guide",
                    "url": "https://docs.fragranceai.com/training"
                }
            },
            {
                "name": "Monitoring",
                "description": "System monitoring and health checks",
                "externalDocs": {
                    "description": "Monitoring Guide",
                    "url": "https://docs.fragranceai.com/monitoring"
                }
            },
            {
                "name": "Admin",
                "description": "Administrative operations (Admin only)",
                "externalDocs": {
                    "description": "Admin Guide",
                    "url": "https://docs.fragranceai.com/admin"
                }
            }
        ]

        # Add response examples
        self._add_response_examples(openapi_schema)

        # Add error codes documentation
        self._add_error_codes(openapi_schema)

        # Add rate limiting information
        self._add_rate_limiting_info(openapi_schema)

        # Add webhooks documentation
        self._add_webhooks_documentation(openapi_schema)

        self.custom_openapi_schema = openapi_schema
        return openapi_schema

    def _get_api_description(self) -> str:
        """Get comprehensive API description"""
        return """
# Fragrance AI API

Welcome to the **Fragrance AI API** - the most advanced AI-powered fragrance creation and discovery platform.

## Overview

The Fragrance AI API enables developers to integrate cutting-edge fragrance intelligence into their applications. Our API provides:

- **🔍 Semantic Search**: Find fragrances using natural language descriptions
- **🧪 AI Generation**: Create unique fragrance recipes using advanced AI models
- **📊 Analytics**: Track user preferences and fragrance trends
- **🔄 Real-time Processing**: Get instant results with optimized performance
- **🛡️ Enterprise Security**: Bank-grade security with rate limiting and monitoring

## Quick Start

1. **Get an API Key**: Sign up at [fragranceai.com](https://fragranceai.com) to get your API key
2. **Authentication**: Include your API key in the `X-API-Key` header
3. **Make Requests**: Start with our search endpoint to explore fragrances
4. **Explore**: Use our interactive documentation to test all endpoints

## Key Features

### 🎯 Semantic Search
```python
# Search for fragrances using natural language
response = requests.post("/api/v1/search/semantic",
    headers={"X-API-Key": "your_key"},
    json={"query": "fresh citrus for summer mornings"})
```

### 🚀 AI Generation
```python
# Generate custom fragrance recipes
response = requests.post("/api/v1/generate/recipe",
    headers={"X-API-Key": "your_key"},
    json={"mood": "romantic", "season": "spring"})
```

### 📈 Real-time Analytics
```python
# Get performance metrics
response = requests.get("/api/v1/monitoring/metrics",
    headers={"X-API-Key": "your_key"})
```

## Rate Limits

- **Free Tier**: 100 requests/hour, 1,000 requests/day
- **Pro Tier**: 1,000 requests/hour, 10,000 requests/day
- **Enterprise**: Custom limits available

## Support

- **Documentation**: [docs.fragranceai.com](https://docs.fragranceai.com)
- **Support**: support@fragranceai.com
- **Status Page**: [status.fragranceai.com](https://status.fragranceai.com)

## SDKs

Official SDKs available for:
- Python: `pip install fragranceai`
- JavaScript: `npm install @fragranceai/sdk`
- Go: `go get github.com/fragranceai/go-sdk`
        """

    def _add_response_examples(self, schema: Dict[str, Any]):
        """Add comprehensive response examples"""

        if "paths" not in schema:
            return

        # Common response examples
        success_examples = {
            "search_response": {
                "summary": "Successful search results",
                "value": {
                    "results": [
                        {
                            "id": "fragrance_001",
                            "name": "Ocean Breeze",
                            "description": "Fresh aquatic fragrance with citrus top notes",
                            "similarity_score": 0.95,
                            "notes": {
                                "top": ["bergamot", "lemon", "sea salt"],
                                "heart": ["jasmine", "lily of the valley"],
                                "base": ["white musk", "cedar"]
                            },
                            "created_at": "2024-01-15T10:30:00Z"
                        }
                    ],
                    "total_count": 1,
                    "query_time_ms": 45,
                    "request_id": "req_123456789"
                }
            },
            "generation_response": {
                "summary": "AI-generated fragrance recipe",
                "value": {
                    "recipe": {
                        "id": "recipe_001",
                        "name": "Mystic Garden",
                        "description": "Enchanting floral blend with mysterious undertones",
                        "ingredients": [
                            {
                                "name": "Rose Bulgarian",
                                "percentage": 25.0,
                                "role": "heart",
                                "supplier": "Givaudan"
                            },
                            {
                                "name": "Bergamot FCF",
                                "percentage": 15.0,
                                "role": "top",
                                "supplier": "IFF"
                            }
                        ],
                        "total_percentage": 100.0,
                        "estimated_cost": 45.50,
                        "complexity_score": 7.8,
                        "created_at": "2024-01-15T10:30:00Z"
                    },
                    "generation_time_ms": 250,
                    "model_version": "v3.2.1",
                    "request_id": "req_987654321"
                }
            }
        }

        error_examples = {
            "validation_error": {
                "summary": "Validation error",
                "value": {
                    "error": "Validation failed",
                    "code": "VALIDATION_ERROR",
                    "details": [
                        {
                            "field": "query",
                            "message": "Query must be between 3 and 500 characters",
                            "invalid_value": "hi"
                        }
                    ],
                    "request_id": "req_error_123"
                }
            },
            "rate_limit_error": {
                "summary": "Rate limit exceeded",
                "value": {
                    "error": "Rate limit exceeded",
                    "code": "RATE_LIMIT_EXCEEDED",
                    "message": "You have exceeded the rate limit of 100 requests per hour",
                    "retry_after": 3600,
                    "request_id": "req_rate_limit_456"
                }
            },
            "authentication_error": {
                "summary": "Authentication failed",
                "value": {
                    "error": "Authentication failed",
                    "code": "INVALID_API_KEY",
                    "message": "The provided API key is invalid or expired",
                    "request_id": "req_auth_789"
                }
            }
        }

        # Add examples to schema
        for path_data in schema["paths"].values():
            for method_data in path_data.values():
                if "responses" in method_data:
                    # Add success examples
                    if "200" in method_data["responses"]:
                        if "search" in str(method_data.get("operationId", "")):
                            method_data["responses"]["200"]["content"] = {
                                "application/json": {
                                    "examples": {"success": success_examples["search_response"]}
                                }
                            }
                        elif "generate" in str(method_data.get("operationId", "")):
                            method_data["responses"]["200"]["content"] = {
                                "application/json": {
                                    "examples": {"success": success_examples["generation_response"]}
                                }
                            }

                    # Add error examples
                    if "400" in method_data["responses"]:
                        method_data["responses"]["400"]["content"] = {
                            "application/json": {
                                "examples": {"validation_error": error_examples["validation_error"]}
                            }
                        }

                    if "401" in method_data["responses"]:
                        method_data["responses"]["401"]["content"] = {
                            "application/json": {
                                "examples": {"auth_error": error_examples["authentication_error"]}
                            }
                        }

                    if "429" in method_data["responses"]:
                        method_data["responses"]["429"]["content"] = {
                            "application/json": {
                                "examples": {"rate_limit": error_examples["rate_limit_error"]}
                            }
                        }

    def _add_error_codes(self, schema: Dict[str, Any]):
        """Add comprehensive error codes documentation"""

        error_codes = {
            "VALIDATION_ERROR": {
                "description": "Request validation failed",
                "http_code": 400,
                "retry": False
            },
            "INVALID_API_KEY": {
                "description": "API key is invalid or expired",
                "http_code": 401,
                "retry": False
            },
            "INSUFFICIENT_PERMISSIONS": {
                "description": "API key lacks required permissions",
                "http_code": 403,
                "retry": False
            },
            "RESOURCE_NOT_FOUND": {
                "description": "Requested resource was not found",
                "http_code": 404,
                "retry": False
            },
            "RATE_LIMIT_EXCEEDED": {
                "description": "Rate limit exceeded",
                "http_code": 429,
                "retry": True
            },
            "INTERNAL_SERVER_ERROR": {
                "description": "Internal server error occurred",
                "http_code": 500,
                "retry": True
            },
            "SERVICE_UNAVAILABLE": {
                "description": "Service temporarily unavailable",
                "http_code": 503,
                "retry": True
            }
        }

        schema["x-error-codes"] = error_codes

    def _add_rate_limiting_info(self, schema: Dict[str, Any]):
        """Add rate limiting documentation"""

        rate_limits = {
            "free": {
                "requests_per_minute": 60,
                "requests_per_hour": 100,
                "requests_per_day": 1000,
                "concurrent_requests": 5
            },
            "pro": {
                "requests_per_minute": 600,
                "requests_per_hour": 1000,
                "requests_per_day": 10000,
                "concurrent_requests": 20
            },
            "enterprise": {
                "requests_per_minute": "unlimited",
                "requests_per_hour": "unlimited",
                "requests_per_day": "unlimited",
                "concurrent_requests": "unlimited"
            }
        }

        schema["x-rate-limits"] = {
            "description": "API rate limits by tier",
            "tiers": rate_limits,
            "headers": {
                "X-RateLimit-Limit": "Request limit per window",
                "X-RateLimit-Remaining": "Requests remaining in current window",
                "X-RateLimit-Reset": "Timestamp when rate limit resets",
                "Retry-After": "Seconds to wait before retrying (when rate limited)"
            }
        }

    def _add_webhooks_documentation(self, schema: Dict[str, Any]):
        """Add webhooks documentation"""

        webhooks = {
            "training_completed": {
                "description": "Triggered when model training completes",
                "method": "POST",
                "headers": {
                    "X-Webhook-Signature": "HMAC-SHA256 signature for verification",
                    "X-Webhook-Event": "training_completed",
                    "Content-Type": "application/json"
                },
                "payload": {
                    "type": "object",
                    "properties": {
                        "event": {"type": "string", "example": "training_completed"},
                        "timestamp": {"type": "string", "format": "date-time"},
                        "data": {
                            "type": "object",
                            "properties": {
                                "training_id": {"type": "string"},
                                "model_version": {"type": "string"},
                                "status": {"type": "string", "enum": ["success", "failed"]},
                                "metrics": {"type": "object"}
                            }
                        }
                    }
                }
            },
            "generation_completed": {
                "description": "Triggered when batch generation completes",
                "method": "POST",
                "headers": {
                    "X-Webhook-Signature": "HMAC-SHA256 signature for verification",
                    "X-Webhook-Event": "generation_completed",
                    "Content-Type": "application/json"
                },
                "payload": {
                    "type": "object",
                    "properties": {
                        "event": {"type": "string", "example": "generation_completed"},
                        "timestamp": {"type": "string", "format": "date-time"},
                        "data": {
                            "type": "object",
                            "properties": {
                                "batch_id": {"type": "string"},
                                "total_recipes": {"type": "integer"},
                                "successful_generations": {"type": "integer"},
                                "failed_generations": {"type": "integer"}
                            }
                        }
                    }
                }
            }
        }

        schema["webhooks"] = webhooks

    def generate_custom_docs_html(self) -> str:
        """Generate custom documentation HTML with enhanced features"""

        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Fragrance AI API Documentation</title>
            <meta charset="utf-8"/>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <link rel="icon" type="image/png" href="/static/favicon.png"/>
            <style>
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    margin: 0;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                }
                .header {
                    background: rgba(255, 255, 255, 0.95);
                    backdrop-filter: blur(10px);
                    padding: 20px;
                    text-align: center;
                    box-shadow: 0 2px 20px rgba(0,0,0,0.1);
                }
                .header h1 {
                    margin: 0;
                    color: #2c3e50;
                    font-size: 2.5em;
                    font-weight: 300;
                }
                .header p {
                    margin: 10px 0 0 0;
                    color: #7f8c8d;
                    font-size: 1.2em;
                }
                .nav {
                    background: rgba(255, 255, 255, 0.9);
                    padding: 15px;
                    text-align: center;
                    border-bottom: 1px solid #ecf0f1;
                }
                .nav a {
                    margin: 0 15px;
                    text-decoration: none;
                    color: #3498db;
                    font-weight: 500;
                    padding: 8px 16px;
                    border-radius: 25px;
                    transition: all 0.3s ease;
                }
                .nav a:hover {
                    background: #3498db;
                    color: white;
                }
                .content {
                    max-width: 1200px;
                    margin: 40px auto;
                    padding: 0 20px;
                }
                .card {
                    background: rgba(255, 255, 255, 0.95);
                    border-radius: 15px;
                    padding: 30px;
                    margin: 20px 0;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                    backdrop-filter: blur(10px);
                }
                .features {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    margin: 30px 0;
                }
                .feature {
                    background: rgba(255, 255, 255, 0.9);
                    padding: 25px;
                    border-radius: 10px;
                    text-align: center;
                    transition: transform 0.3s ease;
                }
                .feature:hover {
                    transform: translateY(-5px);
                }
                .feature-icon {
                    font-size: 3em;
                    margin-bottom: 15px;
                }
                .cta {
                    text-align: center;
                    margin: 40px 0;
                }
                .btn {
                    display: inline-block;
                    padding: 15px 30px;
                    background: linear-gradient(45deg, #3498db, #2980b9);
                    color: white;
                    text-decoration: none;
                    border-radius: 25px;
                    font-weight: 500;
                    margin: 0 10px;
                    transition: all 0.3s ease;
                    box-shadow: 0 5px 15px rgba(52, 152, 219, 0.3);
                }
                .btn:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 7px 20px rgba(52, 152, 219, 0.4);
                }
                .code-example {
                    background: #2c3e50;
                    color: #ecf0f1;
                    padding: 20px;
                    border-radius: 8px;
                    font-family: 'Monaco', 'Courier New', monospace;
                    overflow-x: auto;
                    margin: 15px 0;
                }
                .stats {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin: 30px 0;
                }
                .stat {
                    text-align: center;
                    padding: 20px;
                    background: rgba(255, 255, 255, 0.8);
                    border-radius: 10px;
                }
                .stat-number {
                    font-size: 2.5em;
                    font-weight: bold;
                    color: #3498db;
                }
                .stat-label {
                    color: #7f8c8d;
                    font-size: 0.9em;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🌸 Fragrance AI API</h1>
                <p>The world's most advanced fragrance intelligence platform</p>
            </div>

            <div class="nav">
                <a href="/docs">Interactive Docs</a>
                <a href="/redoc">ReDoc</a>
                <a href="/openapi.json">OpenAPI Spec</a>
                <a href="https://github.com/fragranceai/examples">Examples</a>
                <a href="https://docs.fragranceai.com">Full Guide</a>
            </div>

            <div class="content">
                <div class="card">
                    <h2>🚀 Get Started in Minutes</h2>
                    <p>Integrate powerful fragrance AI into your application with just a few lines of code.</p>

                    <div class="code-example">
curl -X POST "https://api.fragranceai.com/api/v1/search/semantic" \\
  -H "X-API-Key: YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "query": "fresh citrus for summer mornings",
    "top_k": 5
  }'
                    </div>
                </div>

                <div class="features">
                    <div class="feature">
                        <div class="feature-icon">🔍</div>
                        <h3>Semantic Search</h3>
                        <p>Find fragrances using natural language descriptions powered by advanced AI models</p>
                    </div>
                    <div class="feature">
                        <div class="feature-icon">🧪</div>
                        <h3>AI Generation</h3>
                        <p>Create unique fragrance recipes with our state-of-the-art generative AI technology</p>
                    </div>
                    <div class="feature">
                        <div class="feature-icon">📊</div>
                        <h3>Real-time Analytics</h3>
                        <p>Monitor performance and gain insights with comprehensive analytics dashboard</p>
                    </div>
                    <div class="feature">
                        <div class="feature-icon">🛡️</div>
                        <h3>Enterprise Security</h3>
                        <p>Bank-grade security with advanced threat detection and rate limiting</p>
                    </div>
                </div>

                <div class="card">
                    <h2>📈 Platform Statistics</h2>
                    <div class="stats">
                        <div class="stat">
                            <div class="stat-number">10M+</div>
                            <div class="stat-label">API Calls Processed</div>
                        </div>
                        <div class="stat">
                            <div class="stat-number">50K+</div>
                            <div class="stat-label">Unique Fragrances</div>
                        </div>
                        <div class="stat">
                            <div class="stat-number">99.9%</div>
                            <div class="stat-label">Uptime SLA</div>
                        </div>
                        <div class="stat">
                            <div class="stat-number">&lt;100ms</div>
                            <div class="stat-label">Average Response Time</div>
                        </div>
                    </div>
                </div>

                <div class="cta">
                    <a href="/docs" class="btn">🔥 Try Interactive Docs</a>
                    <a href="https://fragranceai.com/signup" class="btn">🚀 Get API Key</a>
                </div>
            </div>
        </body>
        </html>
        """


def setup_documentation_routes(app: FastAPI):
    """Setup enhanced documentation routes"""

    doc_generator = APIDocumentationGenerator(app)

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def documentation_home():
        """Enhanced documentation homepage"""
        return doc_generator.generate_custom_docs_html()

    @app.get("/docs", include_in_schema=False)
    async def custom_swagger_ui_html():
        """Enhanced Swagger UI with custom styling"""
        return get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title=f"{app.title} - Interactive API Documentation",
            oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
            swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4.15.5/swagger-ui-bundle.js",
            swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4.15.5/swagger-ui.css",
            swagger_ui_parameters={
                "deepLinking": True,
                "displayRequestDuration": True,
                "docExpansion": "none",
                "operationsSorter": "alpha",
                "filter": True,
                "showExtensions": True,
                "showCommonExtensions": True,
                "defaultModelsExpandDepth": 3,
                "defaultModelExpandDepth": 3,
                "displayOperationId": True,
                "tryItOutEnabled": True
            }
        )

    @app.get("/redoc", include_in_schema=False)
    async def redoc_html():
        """Enhanced ReDoc documentation"""
        return get_redoc_html(
            openapi_url=app.openapi_url,
            title=f"{app.title} - API Reference",
            redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@2.0.0/bundles/redoc.standalone.js",
            redoc_favicon_url="/static/favicon.png",
            with_google_fonts=True
        )

    @app.get("/openapi.json", include_in_schema=False)
    async def get_enhanced_openapi_schema():
        """Get enhanced OpenAPI schema"""
        return doc_generator.generate_enhanced_openapi_schema()

    @app.get("/api-guide", response_class=HTMLResponse, include_in_schema=False)
    async def api_guide():
        """Comprehensive API usage guide"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Fragrance AI API Guide</title>
            <meta charset="utf-8"/>
            <style>
                body { font-family: -apple-system, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                .guide-section { margin: 30px 0; padding: 20px; border-left: 4px solid #3498db; background: #f8f9fa; }
                code { background: #e9ecef; padding: 2px 6px; border-radius: 3px; }
                pre { background: #2c3e50; color: white; padding: 15px; border-radius: 5px; overflow-x: auto; }
            </style>
        </head>
        <body>
            <h1>Fragrance AI API Usage Guide</h1>

            <div class="guide-section">
                <h2>🔐 Authentication</h2>
                <p>All API requests require authentication using an API key:</p>
                <pre>curl -H "X-API-Key: your_api_key_here" https://api.fragranceai.com/api/v1/search</pre>
            </div>

            <div class="guide-section">
                <h2>🔍 Semantic Search</h2>
                <p>Search for fragrances using natural language:</p>
                <pre>
POST /api/v1/search/semantic
{
  "query": "romantic rose fragrance for evening",
  "top_k": 10,
  "filters": {
    "price_range": [50, 200],
    "brand": ["Chanel", "Dior"]
  }
}
                </pre>
            </div>

            <div class="guide-section">
                <h2>🧪 Recipe Generation</h2>
                <p>Generate custom fragrance recipes:</p>
                <pre>
POST /api/v1/generate/recipe
{
  "mood": "energetic",
  "season": "summer",
  "gender": "unisex",
  "complexity": "medium"
}
                </pre>
            </div>

            <div class="guide-section">
                <h2>📊 Rate Limits</h2>
                <p>API calls are rate limited by tier:</p>
                <ul>
                    <li><strong>Free:</strong> 100 requests/hour</li>
                    <li><strong>Pro:</strong> 1,000 requests/hour</li>
                    <li><strong>Enterprise:</strong> Custom limits</li>
                </ul>
            </div>

            <div class="guide-section">
                <h2>🚨 Error Handling</h2>
                <p>All errors include structured information:</p>
                <pre>
{
  "error": "Validation failed",
  "code": "VALIDATION_ERROR",
  "details": [
    {
      "field": "query",
      "message": "Query is required"
    }
  ],
  "request_id": "req_123456"
}
                </pre>
            </div>
        </body>
        </html>
        """

    # Override the openapi method to use our enhanced schema
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        app.openapi_schema = doc_generator.generate_enhanced_openapi_schema()
        return app.openapi_schema

    app.openapi = custom_openapi

    logger.info("Enhanced API documentation routes configured")


# Export for use in main app
__all__ = ["setup_documentation_routes", "APIDocumentationGenerator"]