@echo off
REM Go / No-Go 배포 게이트 체크 스크립트 (Windows)

echo ==================================================
echo 🚦 Go / No-Go Deployment Gate
echo ==================================================
echo.

REM 환경 변수 설정
set PYTHONPATH=%PYTHONPATH%;%CD%
if "%PROMETHEUS_URL%"=="" set PROMETHEUS_URL=http://localhost:9090

REM Python 실행
python -m fragrance_ai.deployment.go_nogo_gate ^
    --prometheus-url %PROMETHEUS_URL% ^
    --report-file deployment_gate_report.txt ^
    --exit-code

REM Exit code 처리
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ GO: Safe to deploy
    exit /b 0
) else (
    echo.
    echo ⛔ NO-GO: Do not deploy
    echo Check deployment_gate_report.txt for details
    exit /b 1
)
