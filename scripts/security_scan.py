#!/usr/bin/env python3
"""
Automated Security Scanning and Vulnerability Assessment
Comprehensive security audit tool for Fragrance AI
"""

import os
import sys
import json
import subprocess
import asyncio
import requests
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import argparse
import concurrent.futures
from urllib.parse import urljoin

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fragrance_ai.core.production_logging import get_logger
from fragrance_ai.core.config import settings

logger = get_logger(__name__)


@dataclass
class VulnerabilityFinding:
    """Vulnerability finding"""
    severity: str
    category: str
    title: str
    description: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    recommendation: Optional[str] = None
    cve_id: Optional[str] = None
    cvss_score: Optional[float] = None


@dataclass
class SecurityScanResult:
    """Security scan result"""
    scan_type: str
    start_time: datetime
    end_time: datetime
    findings: List[VulnerabilityFinding]
    summary: Dict[str, Any]
    scan_id: str


class CodeSecurityScanner:
    """Static code security analysis"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.findings: List[VulnerabilityFinding] = []

    async def scan_code_quality(self) -> List[VulnerabilityFinding]:
        """Scan code for security issues using bandit"""
        logger.info("Running Bandit security scan...")

        try:
            # Run bandit
            cmd = [
                "bandit", "-r", str(self.project_root / "fragrance_ai"),
                "-f", "json", "-o", "/tmp/bandit_report.json"
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                # Parse bandit report
                if os.path.exists("/tmp/bandit_report.json"):
                    with open("/tmp/bandit_report.json") as f:
                        report = json.load(f)

                    for result in report.get("results", []):
                        finding = VulnerabilityFinding(
                            severity=result.get("issue_severity", "UNKNOWN"),
                            category="SAST",
                            title=result.get("test_name", "Unknown Issue"),
                            description=result.get("issue_text", ""),
                            file_path=result.get("filename", ""),
                            line_number=result.get("line_number", 0),
                            recommendation=result.get("issue_confidence", ""),
                            cve_id=result.get("test_id", "")
                        )
                        self.findings.append(finding)

            return self.findings

        except Exception as e:
            logger.error(f"Bandit scan failed: {e}")
            return []

    async def scan_dependencies(self) -> List[VulnerabilityFinding]:
        """Scan dependencies for known vulnerabilities"""
        logger.info("Running dependency vulnerability scan...")

        try:
            # Run safety check
            cmd = ["safety", "check", "--json"]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if stdout:
                vulnerabilities = json.loads(stdout.decode())

                for vuln in vulnerabilities:
                    finding = VulnerabilityFinding(
                        severity="HIGH" if vuln.get("vulnerability_id") else "MEDIUM",
                        category="DEPENDENCY",
                        title=f"Vulnerable dependency: {vuln.get('package_name', 'Unknown')}",
                        description=vuln.get("advisory", ""),
                        recommendation=f"Update to version {vuln.get('analyzed_version', 'latest')}",
                        cve_id=vuln.get("vulnerability_id", "")
                    )
                    self.findings.append(finding)

            return self.findings

        except Exception as e:
            logger.error(f"Dependency scan failed: {e}")
            return []

    async def scan_secrets(self) -> List[VulnerabilityFinding]:
        """Scan for exposed secrets and credentials"""
        logger.info("Scanning for exposed secrets...")

        # Common secret patterns
        secret_patterns = {
            "API_KEY": r"['\"]?api[_-]?key['\"]?\s*[:=]\s*['\"]?([a-zA-Z0-9_-]{20,})['\"]?",
            "PASSWORD": r"['\"]?password['\"]?\s*[:=]\s*['\"]?([^'\"\\s]{8,})['\"]?",
            "SECRET_KEY": r"['\"]?secret[_-]?key['\"]?\s*[:=]\s*['\"]?([a-zA-Z0-9_-]{20,})['\"]?",
            "JWT_SECRET": r"['\"]?jwt[_-]?secret['\"]?\s*[:=]\s*['\"]?([a-zA-Z0-9_-]{20,})['\"]?",
            "DATABASE_URL": r"['\"]?database[_-]?url['\"]?\s*[:=]\s*['\"]?(postgresql://[^'\"\\s]+)['\"]?",
            "PRIVATE_KEY": r"-----BEGIN [A-Z ]+PRIVATE KEY-----",
            "AWS_ACCESS_KEY": r"AKIA[0-9A-Z]{16}",
            "GITHUB_TOKEN": r"ghp_[a-zA-Z0-9]{36}",
        }

        for file_path in self.project_root.rglob("*.py"):
            if "venv" in str(file_path) or ".git" in str(file_path):
                continue

            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                for secret_type, pattern in secret_patterns.items():
                    import re
                    matches = re.finditer(pattern, content, re.IGNORECASE)

                    for match in matches:
                        # Skip if it looks like a placeholder
                        value = match.group(1) if len(match.groups()) > 0 else match.group(0)
                        if any(placeholder in value.lower() for placeholder in
                               ['your_', 'example', 'placeholder', 'change_this', 'xxx']):
                            continue

                        line_num = content[:match.start()].count('\n') + 1

                        finding = VulnerabilityFinding(
                            severity="CRITICAL",
                            category="SECRET_EXPOSURE",
                            title=f"Potential {secret_type.replace('_', ' ')} exposure",
                            description=f"Found potential {secret_type} in source code",
                            file_path=str(file_path.relative_to(self.project_root)),
                            line_number=line_num,
                            recommendation="Move sensitive data to environment variables or secure vault"
                        )
                        self.findings.append(finding)

            except Exception as e:
                logger.warning(f"Could not scan file {file_path}: {e}")

        return self.findings

    async def scan_file_permissions(self) -> List[VulnerabilityFinding]:
        """Scan for insecure file permissions"""
        logger.info("Scanning file permissions...")

        import stat

        for file_path in self.project_root.rglob("*"):
            if file_path.is_file():
                try:
                    file_stat = file_path.stat()
                    mode = stat.filemode(file_stat.st_mode)

                    # Check for world-writable files
                    if file_stat.st_mode & stat.S_IWOTH:
                        finding = VulnerabilityFinding(
                            severity="MEDIUM",
                            category="FILE_PERMISSIONS",
                            title="World-writable file",
                            description=f"File is writable by all users: {mode}",
                            file_path=str(file_path.relative_to(self.project_root)),
                            recommendation="Restrict file permissions to owner/group only"
                        )
                        self.findings.append(finding)

                    # Check for executable files with write permissions
                    if (file_stat.st_mode & stat.S_IXUSR and
                        file_stat.st_mode & stat.S_IWGRP):
                        finding = VulnerabilityFinding(
                            severity="LOW",
                            category="FILE_PERMISSIONS",
                            title="Executable file with group write",
                            description=f"Executable file writable by group: {mode}",
                            file_path=str(file_path.relative_to(self.project_root)),
                            recommendation="Remove group write permission from executable files"
                        )
                        self.findings.append(finding)

                except Exception as e:
                    continue

        return self.findings


class NetworkSecurityScanner:
    """Network and web application security scanner"""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.findings: List[VulnerabilityFinding] = []

    async def scan_ssl_configuration(self) -> List[VulnerabilityFinding]:
        """Scan SSL/TLS configuration"""
        logger.info("Scanning SSL/TLS configuration...")

        try:
            import ssl
            import socket
            from urllib.parse import urlparse

            parsed_url = urlparse(self.base_url)
            hostname = parsed_url.hostname
            port = parsed_url.port or (443 if parsed_url.scheme == 'https' else 80)

            if parsed_url.scheme != 'https':
                finding = VulnerabilityFinding(
                    severity="HIGH",
                    category="SSL_TLS",
                    title="No HTTPS encryption",
                    description="Application not using HTTPS encryption",
                    recommendation="Enable HTTPS with proper SSL/TLS configuration"
                )
                self.findings.append(finding)
                return self.findings

            # Check SSL certificate
            context = ssl.create_default_context()
            with socket.create_connection((hostname, port), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    cert = ssock.getpeercert()

                    # Check certificate expiration
                    import datetime
                    not_after = datetime.datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
                    days_until_expiry = (not_after - datetime.datetime.now()).days

                    if days_until_expiry < 30:
                        finding = VulnerabilityFinding(
                            severity="HIGH" if days_until_expiry < 7 else "MEDIUM",
                            category="SSL_TLS",
                            title="SSL certificate expires soon",
                            description=f"Certificate expires in {days_until_expiry} days",
                            recommendation="Renew SSL certificate before expiration"
                        )
                        self.findings.append(finding)

                    # Check for weak cipher suites
                    cipher = ssock.cipher()
                    if cipher and len(cipher) > 1:
                        if 'RC4' in cipher[0] or 'DES' in cipher[0]:
                            finding = VulnerabilityFinding(
                                severity="HIGH",
                                category="SSL_TLS",
                                title="Weak cipher suite",
                                description=f"Using weak cipher: {cipher[0]}",
                                recommendation="Disable weak cipher suites"
                            )
                            self.findings.append(finding)

        except Exception as e:
            logger.error(f"SSL scan failed: {e}")

        return self.findings

    async def scan_security_headers(self) -> List[VulnerabilityFinding]:
        """Scan for security headers"""
        logger.info("Scanning security headers...")

        required_headers = {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': ['DENY', 'SAMEORIGIN'],
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': None,  # Just check presence
            'Content-Security-Policy': None,
            'Referrer-Policy': None
        }

        try:
            response = requests.get(self.base_url, timeout=10, verify=False)

            for header, expected_value in required_headers.items():
                header_value = response.headers.get(header)

                if not header_value:
                    finding = VulnerabilityFinding(
                        severity="MEDIUM",
                        category="SECURITY_HEADERS",
                        title=f"Missing security header: {header}",
                        description=f"Response lacks {header} security header",
                        recommendation=f"Add {header} header to all responses"
                    )
                    self.findings.append(finding)
                elif expected_value and isinstance(expected_value, list):
                    if header_value not in expected_value:
                        finding = VulnerabilityFinding(
                            severity="LOW",
                            category="SECURITY_HEADERS",
                            title=f"Weak {header} configuration",
                            description=f"Header value '{header_value}' not optimal",
                            recommendation=f"Use recommended values: {expected_value}"
                        )
                        self.findings.append(finding)
                elif expected_value and expected_value not in header_value:
                    finding = VulnerabilityFinding(
                        severity="LOW",
                        category="SECURITY_HEADERS",
                        title=f"Weak {header} configuration",
                        description=f"Header value '{header_value}' not optimal",
                        recommendation=f"Include '{expected_value}' in header value"
                    )
                    self.findings.append(finding)

        except Exception as e:
            logger.error(f"Header scan failed: {e}")

        return self.findings

    async def scan_endpoints(self) -> List[VulnerabilityFinding]:
        """Scan API endpoints for vulnerabilities"""
        logger.info("Scanning API endpoints...")

        # Common attack payloads
        sql_payloads = [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "UNION SELECT 1,2,3--",
            "' AND (SELECT * FROM (SELECT COUNT(*),CONCAT(version(),FLOOR(RAND(0)*2))x FROM information_schema.tables GROUP BY x)a) --"
        ]

        xss_payloads = [
            "<script>alert('XSS')</script>",
            "javascript:alert(1)",
            "<img src=x onerror=alert(1)>",
            "'\"><script>alert(String.fromCharCode(88,83,83))</script>"
        ]

        # Common endpoints to test
        test_endpoints = [
            "/api/v1/search",
            "/api/v1/generate",
            "/docs",
            "/health",
            "/metrics"
        ]

        for endpoint in test_endpoints:
            url = urljoin(self.base_url, endpoint)

            # Test SQL injection
            for payload in sql_payloads:
                try:
                    response = requests.get(f"{url}?q={payload}", timeout=5)

                    # Look for SQL error messages
                    error_indicators = [
                        "sql", "syntax error", "mysql", "postgresql", "sqlite",
                        "ORA-", "Microsoft SQL", "ODBC", "JDBC"
                    ]

                    response_text = response.text.lower()
                    if any(indicator in response_text for indicator in error_indicators):
                        finding = VulnerabilityFinding(
                            severity="HIGH",
                            category="SQL_INJECTION",
                            title=f"Possible SQL injection in {endpoint}",
                            description=f"Endpoint may be vulnerable to SQL injection",
                            recommendation="Implement parameterized queries and input validation"
                        )
                        self.findings.append(finding)
                        break

                except Exception:
                    continue

            # Test XSS
            for payload in xss_payloads:
                try:
                    response = requests.post(url, json={"query": payload}, timeout=5)

                    if payload in response.text:
                        finding = VulnerabilityFinding(
                            severity="HIGH",
                            category="XSS",
                            title=f"Possible XSS vulnerability in {endpoint}",
                            description="Endpoint reflects user input without proper encoding",
                            recommendation="Implement proper input validation and output encoding"
                        )
                        self.findings.append(finding)
                        break

                except Exception:
                    continue

        return self.findings


class ContainerSecurityScanner:
    """Container and Docker security scanner"""

    def __init__(self, image_name: str):
        self.image_name = image_name
        self.findings: List[VulnerabilityFinding] = []

    async def scan_container_vulnerabilities(self) -> List[VulnerabilityFinding]:
        """Scan container for vulnerabilities using Trivy"""
        logger.info(f"Scanning container image: {self.image_name}")

        try:
            # Run Trivy scan
            cmd = [
                "trivy", "image", "--format", "json",
                "--severity", "HIGH,CRITICAL",
                self.image_name
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if stdout:
                report = json.loads(stdout.decode())

                for result in report.get("Results", []):
                    for vulnerability in result.get("Vulnerabilities", []):
                        finding = VulnerabilityFinding(
                            severity=vulnerability.get("Severity", "UNKNOWN"),
                            category="CONTAINER_VULNERABILITY",
                            title=f"Vulnerable package: {vulnerability.get('PkgName', 'Unknown')}",
                            description=vulnerability.get("Description", ""),
                            recommendation=f"Update to version {vulnerability.get('FixedVersion', 'latest')}",
                            cve_id=vulnerability.get("VulnerabilityID", ""),
                            cvss_score=vulnerability.get("CVSS", {}).get("nvd", {}).get("V3Score", 0.0)
                        )
                        self.findings.append(finding)

        except Exception as e:
            logger.error(f"Container scan failed: {e}")

        return self.findings

    async def scan_dockerfile(self, dockerfile_path: str) -> List[VulnerabilityFinding]:
        """Scan Dockerfile for security issues"""
        logger.info(f"Scanning Dockerfile: {dockerfile_path}")

        if not os.path.exists(dockerfile_path):
            return []

        try:
            with open(dockerfile_path, 'r') as f:
                content = f.read()

            lines = content.split('\n')

            for i, line in enumerate(lines, 1):
                line = line.strip().upper()

                # Check for running as root
                if line.startswith('USER') and 'ROOT' in line:
                    finding = VulnerabilityFinding(
                        severity="HIGH",
                        category="DOCKERFILE",
                        title="Running as root user",
                        description="Container runs as root user",
                        file_path=dockerfile_path,
                        line_number=i,
                        recommendation="Create and use a non-root user"
                    )
                    self.findings.append(finding)

                # Check for latest tag usage
                if 'FROM' in line and ':LATEST' in line:
                    finding = VulnerabilityFinding(
                        severity="MEDIUM",
                        category="DOCKERFILE",
                        title="Using 'latest' tag",
                        description="Base image uses 'latest' tag",
                        file_path=dockerfile_path,
                        line_number=i,
                        recommendation="Use specific version tags for base images"
                    )
                    self.findings.append(finding)

                # Check for ADD instead of COPY
                if line.startswith('ADD ') and not ('http' in line or 'ftp' in line):
                    finding = VulnerabilityFinding(
                        severity="LOW",
                        category="DOCKERFILE",
                        title="Using ADD instead of COPY",
                        description="ADD instruction has implicit behavior",
                        file_path=dockerfile_path,
                        line_number=i,
                        recommendation="Use COPY for local files instead of ADD"
                    )
                    self.findings.append(finding)

        except Exception as e:
            logger.error(f"Dockerfile scan failed: {e}")

        return self.findings


class SecurityAuditor:
    """Main security auditor orchestrator"""

    def __init__(self, project_root: str, base_url: str = None, image_name: str = None):
        self.project_root = project_root
        self.base_url = base_url
        self.image_name = image_name
        self.all_findings: List[VulnerabilityFinding] = []

    async def run_full_audit(self) -> SecurityScanResult:
        """Run complete security audit"""
        scan_id = f"security_audit_{int(time.time())}"
        start_time = datetime.now()

        logger.info(f"Starting comprehensive security audit: {scan_id}")

        # Code security scanning
        code_scanner = CodeSecurityScanner(self.project_root)

        code_tasks = [
            code_scanner.scan_code_quality(),
            code_scanner.scan_dependencies(),
            code_scanner.scan_secrets(),
            code_scanner.scan_file_permissions()
        ]

        # Network security scanning
        network_tasks = []
        if self.base_url:
            network_scanner = NetworkSecurityScanner(self.base_url)
            network_tasks = [
                network_scanner.scan_ssl_configuration(),
                network_scanner.scan_security_headers(),
                network_scanner.scan_endpoints()
            ]

        # Container security scanning
        container_tasks = []
        if self.image_name:
            container_scanner = ContainerSecurityScanner(self.image_name)
            container_tasks = [
                container_scanner.scan_container_vulnerabilities(),
                container_scanner.scan_dockerfile(
                    os.path.join(self.project_root, "docker/Dockerfile.production")
                )
            ]

        # Run all scans concurrently
        all_tasks = code_tasks + network_tasks + container_tasks

        results = await asyncio.gather(*all_tasks, return_exceptions=True)

        # Collect all findings
        for result in results:
            if isinstance(result, list):
                self.all_findings.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Scan task failed: {result}")

        end_time = datetime.now()

        # Generate summary
        summary = self._generate_summary()

        scan_result = SecurityScanResult(
            scan_type="comprehensive",
            start_time=start_time,
            end_time=end_time,
            findings=self.all_findings,
            summary=summary,
            scan_id=scan_id
        )

        logger.info(f"Security audit completed: {len(self.all_findings)} findings")

        return scan_result

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate scan summary"""
        severity_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        category_counts = {}

        for finding in self.all_findings:
            severity_counts[finding.severity] = severity_counts.get(finding.severity, 0) + 1
            category_counts[finding.category] = category_counts.get(finding.category, 0) + 1

        return {
            "total_findings": len(self.all_findings),
            "severity_distribution": severity_counts,
            "category_distribution": category_counts,
            "critical_issues": severity_counts["CRITICAL"],
            "high_issues": severity_counts["HIGH"],
            "risk_score": self._calculate_risk_score(severity_counts)
        }

    def _calculate_risk_score(self, severity_counts: Dict[str, int]) -> float:
        """Calculate overall risk score (0-100)"""
        weights = {"CRITICAL": 10, "HIGH": 5, "MEDIUM": 2, "LOW": 1}

        total_weighted_score = sum(
            severity_counts[severity] * weight
            for severity, weight in weights.items()
        )

        # Normalize to 0-100 scale
        max_possible_score = 100  # Arbitrary max for normalization
        risk_score = min(100, (total_weighted_score / max_possible_score) * 100)

        return round(risk_score, 2)

    def generate_report(self, scan_result: SecurityScanResult, output_file: str):
        """Generate detailed security report"""
        report = {
            "scan_metadata": {
                "scan_id": scan_result.scan_id,
                "scan_type": scan_result.scan_type,
                "start_time": scan_result.start_time.isoformat(),
                "end_time": scan_result.end_time.isoformat(),
                "duration_seconds": (scan_result.end_time - scan_result.start_time).total_seconds()
            },
            "summary": scan_result.summary,
            "findings": [asdict(finding) for finding in scan_result.findings]
        }

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Security report generated: {output_file}")

    def generate_html_report(self, scan_result: SecurityScanResult, output_file: str):
        """Generate HTML security report"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Security Audit Report - {scan_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .critical {{ color: #dc3545; font-weight: bold; }}
                .high {{ color: #fd7e14; font-weight: bold; }}
                .medium {{ color: #ffc107; }}
                .low {{ color: #28a745; }}
                .summary {{ background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                .finding {{ border: 1px solid #dee2e6; margin: 10px 0; padding: 15px; border-radius: 5px; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ border: 1px solid #dee2e6; padding: 8px; text-align: left; }}
                th {{ background-color: #e9ecef; }}
            </style>
        </head>
        <body>
            <h1>Security Audit Report</h1>
            <div class="summary">
                <h2>Executive Summary</h2>
                <p><strong>Scan ID:</strong> {scan_id}</p>
                <p><strong>Total Findings:</strong> {total_findings}</p>
                <p><strong>Risk Score:</strong> {risk_score}/100</p>
                <p><strong>Critical Issues:</strong> <span class="critical">{critical_count}</span></p>
                <p><strong>High Issues:</strong> <span class="high">{high_count}</span></p>
            </div>

            <h2>Findings by Severity</h2>
            <table>
                <tr><th>Severity</th><th>Count</th></tr>
                <tr><td class="critical">CRITICAL</td><td>{critical_count}</td></tr>
                <tr><td class="high">HIGH</td><td>{high_count}</td></tr>
                <tr><td class="medium">MEDIUM</td><td>{medium_count}</td></tr>
                <tr><td class="low">LOW</td><td>{low_count}</td></tr>
            </table>

            <h2>Detailed Findings</h2>
            {findings_html}
        </body>
        </html>
        """

        findings_html = ""
        for finding in scan_result.findings:
            severity_class = finding.severity.lower()
            findings_html += f"""
            <div class="finding">
                <h3 class="{severity_class}">[{finding.severity}] {finding.title}</h3>
                <p><strong>Category:</strong> {finding.category}</p>
                <p><strong>Description:</strong> {finding.description}</p>
                {f'<p><strong>File:</strong> {finding.file_path}:{finding.line_number}</p>' if finding.file_path else ''}
                {f'<p><strong>Recommendation:</strong> {finding.recommendation}</p>' if finding.recommendation else ''}
                {f'<p><strong>CVE:</strong> {finding.cve_id}</p>' if finding.cve_id else ''}
            </div>
            """

        html_content = html_template.format(
            scan_id=scan_result.scan_id,
            total_findings=scan_result.summary["total_findings"],
            risk_score=scan_result.summary["risk_score"],
            critical_count=scan_result.summary["severity_distribution"]["CRITICAL"],
            high_count=scan_result.summary["severity_distribution"]["HIGH"],
            medium_count=scan_result.summary["severity_distribution"]["MEDIUM"],
            low_count=scan_result.summary["severity_distribution"]["LOW"],
            findings_html=findings_html
        )

        with open(output_file, 'w') as f:
            f.write(html_content)

        logger.info(f"HTML security report generated: {output_file}")


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Comprehensive Security Audit")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--base-url", help="Base URL for web application scanning")
    parser.add_argument("--image-name", help="Docker image name for container scanning")
    parser.add_argument("--output", default="security_report.json", help="Output report file")
    parser.add_argument("--html-output", help="HTML output report file")
    parser.add_argument("--scan-type", choices=["code", "network", "container", "full"],
                       default="full", help="Type of scan to perform")

    args = parser.parse_args()

    # Initialize auditor
    auditor = SecurityAuditor(
        project_root=args.project_root,
        base_url=args.base_url,
        image_name=args.image_name
    )

    try:
        # Run audit
        scan_result = await auditor.run_full_audit()

        # Generate reports
        auditor.generate_report(scan_result, args.output)

        if args.html_output:
            auditor.generate_html_report(scan_result, args.html_output)

        # Print summary
        print(f"\nSecurity Audit Summary:")
        print(f"Total Findings: {scan_result.summary['total_findings']}")
        print(f"Risk Score: {scan_result.summary['risk_score']}/100")
        print(f"Critical: {scan_result.summary['critical_issues']}")
        print(f"High: {scan_result.summary['high_issues']}")

        # Exit with error code if critical issues found
        if scan_result.summary['critical_issues'] > 0:
            sys.exit(1)

    except Exception as e:
        logger.error(f"Security audit failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())