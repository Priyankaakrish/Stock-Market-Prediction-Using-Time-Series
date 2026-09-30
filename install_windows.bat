@echo off
REM ════════════════════════════════════════════════════════════════════════
REM  NVDA Stock Prediction — Windows Install Script (Python 3.9 / 3.10 / 3.11)
REM  Run this from the nvda_stock_prediction\ folder:
REM      install_windows.bat
REM ════════════════════════════════════════════════════════════════════════

echo.
echo [1/8] Upgrading pip, setuptools, wheel ...
python -m pip install --upgrade pip setuptools wheel
if %errorlevel% neq 0 ( echo ERROR: pip upgrade failed & pause & exit /b 1 )

echo.
echo [2/8] Installing greenlet pre-built wheel (avoids MSVC C++ error) ...
pip install "greenlet==2.0.2"
if %errorlevel% neq 0 ( echo ERROR: greenlet install failed & pause & exit /b 1 )

echo.
echo [3/8] Installing core data science packages ...
pip install "numpy>=1.24,<2.0" "pandas>=2.0,<3.0" "scipy>=1.11,<2.0" "scikit-learn>=1.3,<2.0" "joblib>=1.3"
if %errorlevel% neq 0 ( echo ERROR: core packages failed & pause & exit /b 1 )

echo.
echo [4/8] Installing time series + gradient boosting ...
pip install "statsmodels>=0.14,<0.15" "pmdarima>=2.0,<3.0" "xgboost>=1.7,<3.0" "lightgbm>=3.3,<5.0" "optuna>=3.3,<4.0" "shap>=0.42,<0.45"
if %errorlevel% neq 0 ( echo ERROR: time series packages failed & pause & exit /b 1 )

echo.
echo [5/8] Installing MLflow (pinned to avoid greenlet conflict) ...
pip install "sqlalchemy>=1.4,<2.0" "mlflow>=2.3,<2.13"
if %errorlevel% neq 0 ( echo ERROR: mlflow install failed & pause & exit /b 1 )

echo.
echo [6/8] Installing API + utilities + visualisation ...
pip install "fastapi>=0.100,<0.115" "uvicorn[standard]>=0.23,<0.30" "pydantic>=2.0,<3.0" "python-multipart>=0.0.6" "httpx>=0.24,<0.28" "tqdm>=4.66" "python-dotenv>=1.0" "pyarrow>=12.0,<17.0" "matplotlib>=3.7,<4.0" "seaborn>=0.12,<0.14" "plotly>=5.15,<6.0" "pytest>=7.4" "pytest-asyncio>=0.21,<0.24"
if %errorlevel% neq 0 ( echo ERROR: API/util packages failed & pause & exit /b 1 )

echo.
echo [7/8] Installing PyTorch (CPU build -- PRIMARY deep-learning backend) ...
echo       This is ~200 MB; may take a minute on slow connections.
pip install torch --index-url https://download.pytorch.org/whl/cpu
if %errorlevel% neq 0 (
    echo WARNING: PyTorch CPU wheel install failed.
    echo          LSTM model will be skipped unless you install PyTorch manually:
    echo            pip install torch --index-url https://download.pytorch.org/whl/cpu
    echo          Continuing without PyTorch ...
) else (
    echo PyTorch installed successfully.
)

echo.
echo [8/8] Verifying key imports ...
python -c "import numpy, pandas, sklearn, mlflow, xgboost, lightgbm; print('Core imports OK')"
python -c "import torch; print('PyTorch', torch.__version__, '-- LSTM backend: torch')" 2>nul || echo "PyTorch not available -- LSTM will be skipped (non-fatal)"

echo.
echo ════════════════════════════════════════════════════════════════════════
echo  Installation complete!
echo.
echo  Optional installs:
echo    Prophet (needs C++ compiler or Python 3.10+):
echo      pip install prophet
echo.
echo  Next steps:
echo    cd nvda_stock_prediction
echo    python pipelines\update_data.py          (optional: latest NVDA prices)
echo    python notebooks\01_EDA_Analysis.py
echo    python pipelines\train_pipeline.py --model xgboost
echo    python pipelines\train_pipeline.py          (all models including LSTM)
echo    uvicorn api.main:app --port 8000
echo ════════════════════════════════════════════════════════════════════════
pause
