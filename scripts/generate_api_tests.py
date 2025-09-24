#!/usr/bin/env python3
"""
Automated API Test Generation
Generates comprehensive test suites from OpenAPI schema
"""

import json
import yaml
import requests
import asyncio
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import argparse
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fragrance_ai.core.production_logging import get_logger

logger = get_logger(__name__)


class APITestGenerator:
    """Generates comprehensive API tests from OpenAPI schema"""

    def __init__(self, base_url: str, openapi_spec: Dict[str, Any]):
        self.base_url = base_url.rstrip('/')
        self.spec = openapi_spec
        self.test_cases = []

    def generate_all_tests(self) -> List[Dict[str, Any]]:
        """Generate comprehensive test suite"""

        logger.info("Generating comprehensive API test suite...")

        # Generate tests for each endpoint
        for path, methods in self.spec.get("paths", {}).items():
            for method, operation in methods.items():
                if method.upper() in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
                    self._generate_endpoint_tests(path, method.upper(), operation)

        # Generate authentication tests
        self._generate_auth_tests()

        # Generate rate limiting tests
        self._generate_rate_limit_tests()

        # Generate error handling tests
        self._generate_error_tests()

        # Generate performance tests
        self._generate_performance_tests()

        logger.info(f"Generated {len(self.test_cases)} test cases")
        return self.test_cases

    def _generate_endpoint_tests(self, path: str, method: str, operation: Dict[str, Any]):
        """Generate tests for a specific endpoint"""

        operation_id = operation.get("operationId", f"{method.lower()}_{path.replace('/', '_')}")

        # Happy path test
        self._add_test_case({
            "name": f"{operation_id}_success",
            "description": f"Test successful {method} {path}",
            "method": method,
            "path": path,
            "expected_status": 200,
            "test_type": "happy_path",
            "request_data": self._generate_valid_request_data(operation),
            "headers": self._generate_auth_headers(),
            "assertions": self._generate_success_assertions(operation)
        })

        # Validation tests
        if method in ["POST", "PUT", "PATCH"]:
            self._generate_validation_tests(path, method, operation)

        # Parameter tests
        if operation.get("parameters"):
            self._generate_parameter_tests(path, method, operation)

        # Response format tests
        self._generate_response_format_tests(path, method, operation)

    def _generate_validation_tests(self, path: str, method: str, operation: Dict[str, Any]):
        """Generate validation tests for request body"""

        operation_id = operation.get("operationId", f"{method.lower()}_{path.replace('/', '_')}")

        # Missing required fields
        self._add_test_case({
            "name": f"{operation_id}_missing_required_fields",
            "description": f"Test {method} {path} with missing required fields",
            "method": method,
            "path": path,
            "expected_status": 400,
            "test_type": "validation",
            "request_data": {},
            "headers": self._generate_auth_headers(),
            "assertions": [
                {"type": "status_code", "expected": 400},
                {"type": "response_contains", "field": "error"},
                {"type": "error_code", "expected": "VALIDATION_ERROR"}
            ]
        })

        # Invalid field types
        invalid_data = self._generate_invalid_request_data(operation)
        if invalid_data:
            self._add_test_case({
                "name": f"{operation_id}_invalid_field_types",
                "description": f"Test {method} {path} with invalid field types",
                "method": method,
                "path": path,
                "expected_status": 400,
                "test_type": "validation",
                "request_data": invalid_data,
                "headers": self._generate_auth_headers(),
                "assertions": [
                    {"type": "status_code", "expected": 400},
                    {"type": "response_contains", "field": "error"}
                ]
            })

        # Boundary value tests
        boundary_data = self._generate_boundary_test_data(operation)
        for boundary_case, data in boundary_data.items():
            self._add_test_case({
                "name": f"{operation_id}_boundary_{boundary_case}",
                "description": f"Test {method} {path} with {boundary_case} values",
                "method": method,
                "path": path,
                "expected_status": 400,
                "test_type": "boundary",
                "request_data": data,
                "headers": self._generate_auth_headers(),
                "assertions": [
                    {"type": "status_code", "expected": 400}
                ]
            })

    def _generate_parameter_tests(self, path: str, method: str, operation: Dict[str, Any]):
        """Generate tests for URL parameters"""

        operation_id = operation.get("operationId", f"{method.lower()}_{path.replace('/', '_')}")

        for param in operation.get("parameters", []):
            if param.get("required"):
                # Test missing required parameter
                self._add_test_case({
                    "name": f"{operation_id}_missing_{param['name']}",
                    "description": f"Test {method} {path} without required parameter {param['name']}",
                    "method": method,
                    "path": path,
                    "expected_status": 400,
                    "test_type": "parameter_validation",
                    "headers": self._generate_auth_headers(),
                    "assertions": [
                        {"type": "status_code", "expected": 400}
                    ]
                })

    def _generate_auth_tests(self):
        """Generate authentication tests"""

        # Test without authentication
        self._add_test_case({
            "name": "authentication_missing_api_key",
            "description": "Test API access without authentication",
            "method": "GET",
            "path": "/api/v1/search/semantic",
            "expected_status": 401,
            "test_type": "authentication",
            "headers": {},
            "assertions": [
                {"type": "status_code", "expected": 401},
                {"type": "error_code", "expected": "INVALID_API_KEY"}
            ]
        })

        # Test with invalid API key
        self._add_test_case({
            "name": "authentication_invalid_api_key",
            "description": "Test API access with invalid API key",
            "method": "GET",
            "path": "/api/v1/search/semantic",
            "expected_status": 401,
            "test_type": "authentication",
            "headers": {"X-API-Key": "invalid_key_12345"},
            "assertions": [
                {"type": "status_code", "expected": 401},
                {"type": "error_code", "expected": "INVALID_API_KEY"}
            ]
        })

        # Test with expired token
        self._add_test_case({
            "name": "authentication_expired_token",
            "description": "Test API access with expired JWT token",
            "method": "GET",
            "path": "/api/v1/admin/users",
            "expected_status": 401,
            "test_type": "authentication",
            "headers": {"Authorization": "Bearer expired.jwt.token"},
            "assertions": [
                {"type": "status_code", "expected": 401}
            ]
        })

    def _generate_rate_limit_tests(self):
        """Generate rate limiting tests"""

        self._add_test_case({
            "name": "rate_limiting_exceeded",
            "description": "Test rate limiting when limits are exceeded",
            "method": "GET",
            "path": "/api/v1/search/semantic",
            "expected_status": 429,
            "test_type": "rate_limiting",
            "headers": self._generate_auth_headers(),
            "special_setup": "make_multiple_requests",
            "setup_params": {"count": 101, "interval": 0.1},
            "assertions": [
                {"type": "status_code", "expected": 429},
                {"type": "error_code", "expected": "RATE_LIMIT_EXCEEDED"},
                {"type": "header_present", "header": "Retry-After"}
            ]
        })

    def _generate_error_tests(self):
        """Generate error handling tests"""

        # Test malformed JSON
        self._add_test_case({
            "name": "error_handling_malformed_json",
            "description": "Test handling of malformed JSON",
            "method": "POST",
            "path": "/api/v1/search/semantic",
            "expected_status": 400,
            "test_type": "error_handling",
            "headers": {**self._generate_auth_headers(), "Content-Type": "application/json"},
            "raw_body": '{"query": "test", "invalid": json}',
            "assertions": [
                {"type": "status_code", "expected": 400}
            ]
        })

        # Test unsupported media type
        self._add_test_case({
            "name": "error_handling_unsupported_media_type",
            "description": "Test unsupported content type",
            "method": "POST",
            "path": "/api/v1/search/semantic",
            "expected_status": 415,
            "test_type": "error_handling",
            "headers": {**self._generate_auth_headers(), "Content-Type": "text/plain"},
            "raw_body": "plain text body",
            "assertions": [
                {"type": "status_code", "expected": 415}
            ]
        })

        # Test method not allowed
        self._add_test_case({
            "name": "error_handling_method_not_allowed",
            "description": "Test method not allowed",
            "method": "DELETE",
            "path": "/api/v1/search/semantic",
            "expected_status": 405,
            "test_type": "error_handling",
            "headers": self._generate_auth_headers(),
            "assertions": [
                {"type": "status_code", "expected": 405}
            ]
        })

    def _generate_performance_tests(self):
        """Generate performance tests"""

        self._add_test_case({
            "name": "performance_search_response_time",
            "description": "Test search endpoint response time",
            "method": "POST",
            "path": "/api/v1/search/semantic",
            "expected_status": 200,
            "test_type": "performance",
            "headers": self._generate_auth_headers(),
            "request_data": {
                "query": "fresh citrus fragrance",
                "top_k": 10
            },
            "assertions": [
                {"type": "status_code", "expected": 200},
                {"type": "response_time_less_than", "max_ms": 500}
            ]
        })

        self._add_test_case({
            "name": "performance_generation_response_time",
            "description": "Test generation endpoint response time",
            "method": "POST",
            "path": "/api/v1/generate/recipe",
            "expected_status": 200,
            "test_type": "performance",
            "headers": self._generate_auth_headers(),
            "request_data": {
                "mood": "romantic",
                "season": "spring"
            },
            "assertions": [
                {"type": "status_code", "expected": 200},
                {"type": "response_time_less_than", "max_ms": 3000}
            ]
        })

    def _generate_response_format_tests(self, path: str, method: str, operation: Dict[str, Any]):
        """Generate response format validation tests"""

        operation_id = operation.get("operationId", f"{method.lower()}_{path.replace('/', '_')}")

        # Test response schema validation
        self._add_test_case({
            "name": f"{operation_id}_response_schema",
            "description": f"Validate response schema for {method} {path}",
            "method": method,
            "path": path,
            "expected_status": 200,
            "test_type": "response_validation",
            "headers": self._generate_auth_headers(),
            "request_data": self._generate_valid_request_data(operation),
            "assertions": [
                {"type": "status_code", "expected": 200},
                {"type": "response_schema_valid"},
                {"type": "required_fields_present"}
            ]
        })

    def _generate_valid_request_data(self, operation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate valid request data based on operation schema"""

        # This is a simplified example - in practice, you'd parse the OpenAPI schema
        path = operation.get("operationId", "")

        if "search" in path:
            return {
                "query": "romantic rose fragrance for spring evenings",
                "top_k": 10,
                "search_type": "similarity"
            }
        elif "generate" in path:
            return {
                "mood": "romantic",
                "season": "spring",
                "family": "floral",
                "intensity": "moderate"
            }
        elif "train" in path:
            return {
                "dataset_path": "data/training/test_dataset.json",
                "model_type": "embedding",
                "epochs": 3
            }

        return {}

    def _generate_invalid_request_data(self, operation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate invalid request data for validation testing"""

        path = operation.get("operationId", "")

        if "search" in path:
            return {
                "query": 123,  # Invalid type - should be string
                "top_k": "invalid",  # Invalid type - should be int
                "search_type": "invalid_type"  # Invalid enum value
            }
        elif "generate" in path:
            return {
                "mood": "invalid_mood",
                "season": 123,
                "intensity": None
            }

        return None

    def _generate_boundary_test_data(self, operation: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Generate boundary value test data"""

        path = operation.get("operationId", "")
        boundary_cases = {}

        if "search" in path:
            boundary_cases.update({
                "query_too_short": {"query": "ab", "top_k": 10},
                "query_too_long": {"query": "x" * 1000, "top_k": 10},
                "top_k_zero": {"query": "test", "top_k": 0},
                "top_k_too_large": {"query": "test", "top_k": 1000}
            })

        return boundary_cases

    def _generate_auth_headers(self) -> Dict[str, str]:
        """Generate authentication headers"""
        return {
            "X-API-Key": "test_api_key_12345",
            "Content-Type": "application/json"
        }

    def _generate_success_assertions(self, operation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate success assertions for operation"""

        base_assertions = [
            {"type": "status_code", "expected": 200},
            {"type": "response_time_less_than", "max_ms": 5000},
            {"type": "response_has_json"}
        ]

        path = operation.get("operationId", "")

        if "search" in path:
            base_assertions.extend([
                {"type": "field_present", "field": "results"},
                {"type": "field_present", "field": "total_count"},
                {"type": "field_present", "field": "request_id"},
                {"type": "field_type", "field": "results", "expected_type": "list"}
            ])
        elif "generate" in path:
            base_assertions.extend([
                {"type": "field_present", "field": "recipe"},
                {"type": "field_present", "field": "generation_time_ms"},
                {"type": "field_present", "field": "model_version"}
            ])

        return base_assertions

    def _add_test_case(self, test_case: Dict[str, Any]):
        """Add a test case to the collection"""
        test_case["id"] = f"test_{len(self.test_cases):04d}"
        test_case["created_at"] = datetime.now().isoformat()
        self.test_cases.append(test_case)

    def export_postman_collection(self, output_file: str):
        """Export tests as Postman collection"""

        collection = {
            "info": {
                "name": "Fragrance AI API Tests",
                "description": "Comprehensive test suite for Fragrance AI API",
                "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json",
                "version": "1.0.0"
            },
            "variable": [
                {
                    "key": "base_url",
                    "value": self.base_url,
                    "type": "string"
                },
                {
                    "key": "api_key",
                    "value": "{{API_KEY}}",
                    "type": "string"
                }
            ],
            "item": []
        }

        for test in self.test_cases:
            item = {
                "name": test["name"],
                "request": {
                    "method": test["method"],
                    "header": [
                        {
                            "key": key,
                            "value": value,
                            "type": "text"
                        }
                        for key, value in test.get("headers", {}).items()
                    ],
                    "url": {
                        "raw": f"{{{{base_url}}}}{test['path']}",
                        "host": ["{{base_url}}"],
                        "path": test["path"].strip("/").split("/")
                    }
                },
                "event": [
                    {
                        "listen": "test",
                        "script": {
                            "exec": self._generate_postman_test_script(test),
                            "type": "text/javascript"
                        }
                    }
                ]
            }

            if test.get("request_data"):
                item["request"]["body"] = {
                    "mode": "raw",
                    "raw": json.dumps(test["request_data"], indent=2),
                    "options": {
                        "raw": {
                            "language": "json"
                        }
                    }
                }

            collection["item"].append(item)

        with open(output_file, 'w') as f:
            json.dump(collection, f, indent=2)

        logger.info(f"Postman collection exported to {output_file}")

    def _generate_postman_test_script(self, test: Dict[str, Any]) -> List[str]:
        """Generate Postman test script for assertions"""

        script_lines = [
            f"// Test: {test['description']}",
            "pm.test('Response time is acceptable', function () {",
            "    pm.expect(pm.response.responseTime).to.be.below(5000);",
            "});",
            ""
        ]

        for assertion in test.get("assertions", []):
            if assertion["type"] == "status_code":
                script_lines.extend([
                    f"pm.test('Status code is {assertion['expected']}', function () {{",
                    f"    pm.response.to.have.status({assertion['expected']});",
                    "});",
                    ""
                ])
            elif assertion["type"] == "field_present":
                script_lines.extend([
                    f"pm.test('Response has {assertion['field']} field', function () {{",
                    "    const jsonData = pm.response.json();",
                    f"    pm.expect(jsonData).to.have.property('{assertion['field']}');",
                    "});",
                    ""
                ])
            elif assertion["type"] == "error_code":
                script_lines.extend([
                    f"pm.test('Error code is {assertion['expected']}', function () {{",
                    "    const jsonData = pm.response.json();",
                    f"    pm.expect(jsonData.code).to.eql('{assertion['expected']}');",
                    "});",
                    ""
                ])

        return script_lines

    def export_pytest_tests(self, output_file: str):
        """Export tests as pytest test file"""

        pytest_content = '''"""
Auto-generated API tests
Generated from OpenAPI specification
"""

import pytest
import requests
import time
import json
from typing import Dict, Any

BASE_URL = "''' + self.base_url + '''"
API_KEY = "test_api_key_12345"

class TestAPI:
    """Comprehensive API test suite"""

    @pytest.fixture
    def auth_headers(self):
        """Authentication headers"""
        return {
            "X-API-Key": API_KEY,
            "Content-Type": "application/json"
        }

    @pytest.fixture
    def client(self):
        """HTTP client session"""
        session = requests.Session()
        session.headers.update({"X-API-Key": API_KEY})
        return session

'''

        for test in self.test_cases:
            pytest_content += self._generate_pytest_method(test)

        with open(output_file, 'w') as f:
            f.write(pytest_content)

        logger.info(f"Pytest tests exported to {output_file}")

    def _generate_pytest_method(self, test: Dict[str, Any]) -> str:
        """Generate pytest method for a test case"""

        method_name = test["name"].replace("-", "_")

        method_code = f'''
    def test_{method_name}(self, client):
        """
        {test['description']}
        Type: {test['test_type']}
        """
        url = f"{{BASE_URL}}{test['path']}"

'''

        if test.get("request_data"):
            method_code += f'''        data = {json.dumps(test['request_data'], indent=8)}

        response = client.{test['method'].lower()}(url, json=data)
'''
        else:
            method_code += f'''        response = client.{test['method'].lower()}(url)
'''

        # Add assertions
        for assertion in test.get("assertions", []):
            if assertion["type"] == "status_code":
                method_code += f'''        assert response.status_code == {assertion['expected']}
'''
            elif assertion["type"] == "field_present":
                method_code += f'''        assert "{assertion['field']}" in response.json()
'''
            elif assertion["type"] == "response_time_less_than":
                method_code += f'''        assert response.elapsed.total_seconds() * 1000 < {assertion['max_ms']}
'''

        return method_code


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate API tests from OpenAPI spec")
    parser.add_argument("--base-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--spec-url", help="OpenAPI spec URL")
    parser.add_argument("--spec-file", help="Local OpenAPI spec file")
    parser.add_argument("--output-dir", default="./generated_tests", help="Output directory")
    parser.add_argument("--format", choices=["postman", "pytest", "both"], default="both", help="Output format")

    args = parser.parse_args()

    # Load OpenAPI spec
    if args.spec_url:
        logger.info(f"Fetching OpenAPI spec from {args.spec_url}")
        response = requests.get(args.spec_url)
        spec = response.json()
    elif args.spec_file:
        logger.info(f"Loading OpenAPI spec from {args.spec_file}")
        with open(args.spec_file) as f:
            if args.spec_file.endswith('.yaml') or args.spec_file.endswith('.yml'):
                spec = yaml.safe_load(f)
            else:
                spec = json.load(f)
    else:
        # Try to fetch from default endpoint
        spec_url = f"{args.base_url}/openapi.json"
        logger.info(f"Fetching OpenAPI spec from {spec_url}")
        try:
            response = requests.get(spec_url)
            spec = response.json()
        except Exception as e:
            logger.error(f"Could not fetch OpenAPI spec: {e}")
            return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Generate tests
    generator = APITestGenerator(args.base_url, spec)
    test_cases = generator.generate_all_tests()

    # Export in requested formats
    if args.format in ["postman", "both"]:
        postman_file = output_dir / "fragrance_ai_tests.postman_collection.json"
        generator.export_postman_collection(str(postman_file))

    if args.format in ["pytest", "both"]:
        pytest_file = output_dir / "test_api_generated.py"
        generator.export_pytest_tests(str(pytest_file))

    # Generate test summary
    summary = {
        "total_tests": len(test_cases),
        "test_types": {},
        "generated_at": datetime.now().isoformat(),
        "base_url": args.base_url
    }

    for test in test_cases:
        test_type = test.get("test_type", "unknown")
        summary["test_types"][test_type] = summary["test_types"].get(test_type, 0) + 1

    summary_file = output_dir / "test_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Generated {len(test_cases)} tests")
    logger.info(f"Output directory: {output_dir}")
    print(f"\n✅ Test generation complete!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Total tests: {len(test_cases)}")
    print(f"📋 Test types: {summary['test_types']}")


if __name__ == "__main__":
    asyncio.run(main())