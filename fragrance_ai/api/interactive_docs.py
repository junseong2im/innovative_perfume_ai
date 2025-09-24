"""
Interactive API Documentation System
Live examples, code generation, and real-time testing
"""

import json
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import aiofiles

from ..core.production_logging import get_logger
from ..core.config import settings

logger = get_logger(__name__)

# Templates directory
templates = Jinja2Templates(directory="templates")


class InteractiveDocumentationGenerator:
    """Advanced interactive documentation system"""

    def __init__(self, app: FastAPI):
        self.app = app
        self.active_connections: List[WebSocket] = []
        self.example_cache: Dict[str, Any] = {}
        self.setup_routes()

    def setup_routes(self):
        """Setup interactive documentation routes"""

        @self.app.websocket("/ws/docs")
        async def websocket_endpoint(websocket: WebSocket):
            await self.connect(websocket)
            try:
                while True:
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    await self.handle_websocket_message(websocket, message)
            except WebSocketDisconnect:
                self.disconnect(websocket)

        @self.app.get("/interactive-docs", response_class=HTMLResponse)
        async def interactive_docs(request: Request):
            """Interactive documentation page"""
            return templates.TemplateResponse("interactive_docs.html", {
                "request": request,
                "title": "Fragrance AI - Interactive API Documentation",
                "base_url": str(request.base_url),
                "endpoints": await self.get_endpoint_metadata()
            })

        @self.app.get("/api-playground", response_class=HTMLResponse)
        async def api_playground(request: Request):
            """API playground for testing"""
            return templates.TemplateResponse("api_playground.html", {
                "request": request,
                "title": "Fragrance AI - API Playground",
                "base_url": str(request.base_url)
            })

        @self.app.get("/code-examples/{language}/{endpoint_id}")
        async def get_code_example(language: str, endpoint_id: str):
            """Generate code examples for specific language and endpoint"""
            return await self.generate_code_example(language, endpoint_id)

        @self.app.post("/test-endpoint")
        async def test_endpoint(request: Request):
            """Test an API endpoint with provided parameters"""
            data = await request.json()
            return await self.execute_test_request(data)

        @self.app.get("/example-data/{endpoint_id}")
        async def get_example_data(endpoint_id: str):
            """Get example request/response data for endpoint"""
            return await self.get_endpoint_examples(endpoint_id)

    async def connect(self, websocket: WebSocket):
        """Connect websocket client"""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("WebSocket client connected to documentation system")

    def disconnect(self, websocket: WebSocket):
        """Disconnect websocket client"""
        self.active_connections.remove(websocket)
        logger.info("WebSocket client disconnected from documentation system")

    async def handle_websocket_message(self, websocket: WebSocket, message: Dict[str, Any]):
        """Handle websocket messages"""
        message_type = message.get("type")

        if message_type == "test_request":
            result = await self.execute_live_test(message["data"])
            await websocket.send_text(json.dumps({
                "type": "test_result",
                "data": result
            }))

        elif message_type == "generate_code":
            code = await self.generate_live_code(message["data"])
            await websocket.send_text(json.dumps({
                "type": "code_generated",
                "data": code
            }))

        elif message_type == "get_examples":
            examples = await self.get_live_examples(message["data"])
            await websocket.send_text(json.dumps({
                "type": "examples",
                "data": examples
            }))

    async def get_endpoint_metadata(self) -> List[Dict[str, Any]]:
        """Get metadata for all API endpoints"""

        endpoints = []

        # Search endpoints
        endpoints.append({
            "id": "semantic_search",
            "name": "Semantic Search",
            "method": "POST",
            "path": "/api/v1/search/semantic",
            "description": "Search for fragrances using natural language descriptions",
            "category": "Search",
            "complexity": "beginner",
            "response_time": "~200ms",
            "example_request": {
                "query": "romantic rose fragrance for spring evenings",
                "top_k": 10,
                "filters": {
                    "families": ["floral"],
                    "seasons": ["spring"],
                    "price_range": [50, 200]
                }
            },
            "use_cases": [
                "Find fragrances by description",
                "Discover similar products",
                "Filter by preferences"
            ]
        })

        # Generation endpoints
        endpoints.append({
            "id": "generate_recipe",
            "name": "Generate Recipe",
            "method": "POST",
            "path": "/api/v1/generate/recipe",
            "description": "Generate custom fragrance recipes using AI",
            "category": "Generation",
            "complexity": "intermediate",
            "response_time": "~2s",
            "example_request": {
                "mood": "romantic",
                "season": "spring",
                "family": "floral",
                "intensity": "moderate",
                "inspiration": "A garden party on a sunny afternoon"
            },
            "use_cases": [
                "Create unique fragrances",
                "Prototype new products",
                "Generate variations"
            ]
        })

        # Training endpoints
        endpoints.append({
            "id": "train_model",
            "name": "Train Model",
            "method": "POST",
            "path": "/api/v1/training/start",
            "description": "Start training custom AI models",
            "category": "Training",
            "complexity": "advanced",
            "response_time": "~30min",
            "example_request": {
                "model_type": "embedding",
                "dataset_path": "data/custom_fragrances.json",
                "training_config": {
                    "epochs": 10,
                    "batch_size": 32,
                    "learning_rate": 1e-4
                }
            },
            "use_cases": [
                "Custom model training",
                "Domain adaptation",
                "Performance optimization"
            ]
        })

        return endpoints

    async def generate_code_example(self, language: str, endpoint_id: str) -> Dict[str, Any]:
        """Generate code examples for specific language and endpoint"""

        endpoint_examples = {
            "semantic_search": {
                "python": {
                    "requests": '''import requests

# Search for fragrances
url = "https://api.fragranceai.com/api/v1/search/semantic"
headers = {
    "X-API-Key": "your_api_key_here",
    "Content-Type": "application/json"
}

data = {
    "query": "romantic rose fragrance for spring evenings",
    "top_k": 10,
    "filters": {
        "families": ["floral"],
        "seasons": ["spring"],
        "price_range": [50, 200]
    }
}

response = requests.post(url, headers=headers, json=data)
results = response.json()

print(f"Found {len(results['results'])} fragrances")
for fragrance in results['results'][:3]:
    print(f"- {fragrance['name']} (Score: {fragrance['similarity_score']:.2f})")''',

                    "aiohttp": '''import aiohttp
import asyncio

async def search_fragrances():
    url = "https://api.fragranceai.com/api/v1/search/semantic"
    headers = {
        "X-API-Key": "your_api_key_here",
        "Content-Type": "application/json"
    }

    data = {
        "query": "romantic rose fragrance for spring evenings",
        "top_k": 10
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=data) as response:
            results = await response.json()
            return results

# Run the search
results = asyncio.run(search_fragrances())
print(f"Found {len(results['results'])} fragrances")''',

                    "sdk": '''from fragranceai import FragranceAI

# Initialize client
client = FragranceAI(api_key="your_api_key_here")

# Search for fragrances
results = client.search.semantic(
    query="romantic rose fragrance for spring evenings",
    top_k=10,
    filters={
        "families": ["floral"],
        "seasons": ["spring"]
    }
)

print(f"Found {len(results.fragrances)} fragrances")
for fragrance in results.fragrances[:3]:
    print(f"- {fragrance.name} (Score: {fragrance.similarity_score:.2f})")'''
                },

                "javascript": {
                    "fetch": '''// Search for fragrances using fetch
const searchFragrances = async () => {
    const url = "https://api.fragranceai.com/api/v1/search/semantic";

    const response = await fetch(url, {
        method: "POST",
        headers: {
            "X-API-Key": "your_api_key_here",
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            query: "romantic rose fragrance for spring evenings",
            top_k: 10,
            filters: {
                families: ["floral"],
                seasons: ["spring"],
                price_range: [50, 200]
            }
        })
    });

    const results = await response.json();

    console.log(`Found ${results.results.length} fragrances`);
    results.results.slice(0, 3).forEach(fragrance => {
        console.log(`- ${fragrance.name} (Score: ${fragrance.similarity_score.toFixed(2)})`);
    });
};

searchFragrances();''',

                    "axios": '''const axios = require('axios');

// Search for fragrances using axios
const searchFragrances = async () => {
    try {
        const response = await axios.post(
            'https://api.fragranceai.com/api/v1/search/semantic',
            {
                query: 'romantic rose fragrance for spring evenings',
                top_k: 10,
                filters: {
                    families: ['floral'],
                    seasons: ['spring']
                }
            },
            {
                headers: {
                    'X-API-Key': 'your_api_key_here',
                    'Content-Type': 'application/json'
                }
            }
        );

        const results = response.data;
        console.log(`Found ${results.results.length} fragrances`);

        results.results.slice(0, 3).forEach(fragrance => {
            console.log(`- ${fragrance.name} (Score: ${fragrance.similarity_score.toFixed(2)})`);
        });

    } catch (error) {
        console.error('Error searching fragrances:', error.response?.data || error.message);
    }
};

searchFragrances();''',

                    "sdk": '''const { FragranceAI } = require('@fragranceai/sdk');

// Initialize client
const client = new FragranceAI({
    apiKey: 'your_api_key_here'
});

// Search for fragrances
const searchFragrances = async () => {
    try {
        const results = await client.search.semantic({
            query: 'romantic rose fragrance for spring evenings',
            topK: 10,
            filters: {
                families: ['floral'],
                seasons: ['spring']
            }
        });

        console.log(`Found ${results.fragrances.length} fragrances`);
        results.fragrances.slice(0, 3).forEach(fragrance => {
            console.log(`- ${fragrance.name} (Score: ${fragrance.similarityScore.toFixed(2)})`);
        });

    } catch (error) {
        console.error('Search failed:', error.message);
    }
};

searchFragrances();'''
                },

                "curl": '''# Search for fragrances using cURL
curl -X POST "https://api.fragranceai.com/api/v1/search/semantic" \\
  -H "X-API-Key: your_api_key_here" \\
  -H "Content-Type: application/json" \\
  -d '{
    "query": "romantic rose fragrance for spring evenings",
    "top_k": 10,
    "filters": {
      "families": ["floral"],
      "seasons": ["spring"],
      "price_range": [50, 200]
    }
  }' | jq '.'

# Extract just the fragrance names and scores
curl -X POST "https://api.fragranceai.com/api/v1/search/semantic" \\
  -H "X-API-Key: your_api_key_here" \\
  -H "Content-Type: application/json" \\
  -d '{
    "query": "romantic rose fragrance for spring evenings",
    "top_k": 5
  }' | jq '.results[] | {name: .name, score: .similarity_score}' ''',

                "go": '''package main

import (
    "bytes"
    "encoding/json"
    "fmt"
    "io"
    "net/http"
)

type SearchRequest struct {
    Query   string                 `json:"query"`
    TopK    int                   `json:"top_k"`
    Filters map[string]interface{} `json:"filters,omitempty"`
}

type SearchResponse struct {
    Results []struct {
        Name            string  `json:"name"`
        SimilarityScore float64 `json:"similarity_score"`
    } `json:"results"`
}

func searchFragrances() error {
    url := "https://api.fragranceai.com/api/v1/search/semantic"

    request := SearchRequest{
        Query: "romantic rose fragrance for spring evenings",
        TopK:  10,
        Filters: map[string]interface{}{
            "families": []string{"floral"},
            "seasons":  []string{"spring"},
        },
    }

    requestBytes, err := json.Marshal(request)
    if err != nil {
        return err
    }

    req, err := http.NewRequest("POST", url, bytes.NewBuffer(requestBytes))
    if err != nil {
        return err
    }

    req.Header.Set("X-API-Key", "your_api_key_here")
    req.Header.Set("Content-Type", "application/json")

    client := &http.Client{}
    resp, err := client.Do(req)
    if err != nil {
        return err
    }
    defer resp.Body.Close()

    body, err := io.ReadAll(resp.Body)
    if err != nil {
        return err
    }

    var response SearchResponse
    if err := json.Unmarshal(body, &response); err != nil {
        return err
    }

    fmt.Printf("Found %d fragrances\\n", len(response.Results))
    for i, fragrance := range response.Results[:3] {
        fmt.Printf("- %s (Score: %.2f)\\n", fragrance.Name, fragrance.SimilarityScore)
    }

    return nil
}

func main() {
    if err := searchFragrances(); err != nil {
        fmt.Printf("Error: %v\\n", err)
    }
}'''
            }
        }

        return endpoint_examples.get(endpoint_id, {}).get(language, {})

    async def execute_test_request(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a test API request"""

        try:
            import aiohttp

            method = test_data.get("method", "GET").upper()
            path = test_data.get("path", "/")
            headers = test_data.get("headers", {})
            body = test_data.get("body", {})

            url = f"{settings.base_url}{path}"

            async with aiohttp.ClientSession() as session:
                if method == "GET":
                    async with session.get(url, headers=headers) as response:
                        result = await self._process_response(response)
                elif method == "POST":
                    async with session.post(url, headers=headers, json=body) as response:
                        result = await self._process_response(response)
                else:
                    result = {"error": f"Method {method} not supported in test mode"}

            return result

        except Exception as e:
            return {
                "error": f"Test execution failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    async def _process_response(self, response) -> Dict[str, Any]:
        """Process HTTP response for testing"""

        try:
            response_data = await response.json()
        except:
            response_data = await response.text()

        return {
            "status_code": response.status,
            "headers": dict(response.headers),
            "body": response_data,
            "success": 200 <= response.status < 300,
            "timestamp": datetime.now().isoformat()
        }

    async def execute_live_test(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute live test via websocket"""
        return await self.execute_test_request(test_data)

    async def generate_live_code(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate code examples in real-time"""

        language = request_data.get("language", "python")
        endpoint_id = request_data.get("endpoint_id", "semantic_search")
        method = request_data.get("method", "requests")

        examples = await self.generate_code_example(language, endpoint_id)

        return {
            "language": language,
            "method": method,
            "code": examples.get(method, "// No example available"),
            "endpoint_id": endpoint_id
        }

    async def get_live_examples(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get live examples for endpoint"""

        endpoint_id = request_data.get("endpoint_id")
        examples = await self.get_endpoint_examples(endpoint_id)

        return examples

    async def get_endpoint_examples(self, endpoint_id: str) -> Dict[str, Any]:
        """Get example data for specific endpoint"""

        examples = {
            "semantic_search": {
                "request_examples": [
                    {
                        "name": "Basic Search",
                        "description": "Simple fragrance search",
                        "data": {
                            "query": "fresh citrus fragrance",
                            "top_k": 5
                        }
                    },
                    {
                        "name": "Advanced Search",
                        "description": "Search with filters and preferences",
                        "data": {
                            "query": "romantic rose fragrance for date night",
                            "top_k": 10,
                            "filters": {
                                "families": ["floral", "oriental"],
                                "seasons": ["spring", "summer"],
                                "price_range": [50, 300],
                                "intensity": ["moderate", "strong"]
                            },
                            "min_similarity_score": 0.7
                        }
                    },
                    {
                        "name": "Mood-based Search",
                        "description": "Search by mood and occasion",
                        "data": {
                            "query": "energizing morning fragrance for office",
                            "top_k": 8,
                            "filters": {
                                "mood": ["energetic", "fresh"],
                                "gender": "unisex"
                            }
                        }
                    }
                ],
                "response_example": {
                    "results": [
                        {
                            "id": "fragrance_001",
                            "name": "Ocean Breeze",
                            "brand": "Aqua Luxury",
                            "similarity_score": 0.92,
                            "description": "Fresh aquatic fragrance with citrus top notes",
                            "price": 150.00,
                            "profile": {
                                "family": "fresh",
                                "notes": {
                                    "top": ["bergamot", "lemon", "sea salt"],
                                    "heart": ["jasmine", "lily of the valley"],
                                    "base": ["white musk", "cedar"]
                                }
                            }
                        }
                    ],
                    "total_count": 1,
                    "query_time_ms": 45.2,
                    "request_id": "req_123456789"
                }
            },

            "generate_recipe": {
                "request_examples": [
                    {
                        "name": "Romantic Floral",
                        "description": "Generate romantic floral fragrance",
                        "data": {
                            "mood": "romantic",
                            "family": "floral",
                            "season": "spring",
                            "intensity": "moderate",
                            "inspiration": "A walk through a blooming rose garden"
                        }
                    },
                    {
                        "name": "Summer Fresh",
                        "description": "Create energizing summer fragrance",
                        "data": {
                            "mood": "energetic",
                            "family": "citrus",
                            "season": "summer",
                            "intensity": "light",
                            "target_price": 80.00,
                            "must_include_notes": ["lemon", "mint"]
                        }
                    }
                ],
                "response_example": {
                    "recipe": {
                        "id": "recipe_001",
                        "name": "Rose Garden Dreams",
                        "description": "Elegant floral composition with romantic undertones",
                        "ingredients": [
                            {
                                "name": "Rose Bulgarian",
                                "percentage": 25.0,
                                "position": "heart",
                                "cost_per_kg": 1250.00
                            },
                            {
                                "name": "Bergamot FCF",
                                "percentage": 20.0,
                                "position": "top",
                                "cost_per_kg": 85.00
                            }
                        ],
                        "estimated_cost": 95.50,
                        "confidence_score": 0.89
                    },
                    "generation_time_ms": 1250.5,
                    "model_version": "v3.2.1"
                }
            }
        }

        return examples.get(endpoint_id, {})


def setup_interactive_docs(app: FastAPI):
    """Setup interactive documentation system"""

    interactive_system = InteractiveDocumentationSystem(app)

    logger.info("Interactive documentation system initialized")

    return interactive_system


# Export for main app
__all__ = ["setup_interactive_docs", "InteractiveDocumentationSystem"]