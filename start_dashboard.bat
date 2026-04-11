@echo off
echo ==========================================
echo   Donor Insights Dashboard Setup
echo ==========================================
echo.
echo [1/3] Generating fresh dataset...
python src/generate_dataset.py
if %errorlevel% neq 0 (
    echo Error generating dataset!
    pause
    exit /b
)

echo.
echo [2/3] Analyzing data and updating website stats...
python notebooks/analysis.py
if %errorlevel% neq 0 (
    echo Error analyzing data!
    pause
    exit /b
)

echo.
echo [3/3] Starting Local Dashboard Server...
echo The dashboard will be available at http://localhost:8000
echo Press Ctrl+C to stop the server.
echo.
cd website
python -m http.server 8000
