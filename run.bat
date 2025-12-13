@echo off
chcp 65001 >nul
title Film Negative Processor
echo ========================================
echo   Film Negative Real-time Processor v1.0
echo ========================================
echo.
echo [INFO] Configuration detected:
echo   - Virtual environment: tube
echo   - Main program: Aurhythm.py
echo.

REM 切换到脚本所在目录
cd /d "%~dp0"

REM 检查主程序文件
if not exist "Aurhythm.py" (
    echo [ERROR] Aurhythm.py not found in current directory!
    echo Please ensure the file exists.
    pause
    exit /b 1
)

REM 激活虚拟环境 tube
echo [INFO] Activating virtual environment: tube
if exist "tube\Scripts\activate.bat" (
    call "tube\Scripts\activate.bat"
) else (
    echo [ERROR] Virtual environment 'tube' not found!
    echo Expected path: %CD%\tube\Scripts\activate.bat
    echo.
    echo If your venv is named differently, please edit this script.
    pause
    exit /b 1
)

REM 检查虚拟环境中的Python
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python not found in virtual environment!
    pause
    exit /b 1
)

REM 显示Python信息
for /f "tokens=*" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [INFO] Using %PYTHON_VERSION% from virtual environment
echo.

REM 检查虚拟环境中是否已安装必要依赖
echo [INFO] Checking dependencies in virtual environment...
python -c "
try:
    import colour, numpy, imageio, PIL, matplotlib
    print('[OK] All required libraries are installed')
except ImportError as e:
    print('[ERROR] Missing library:', e)
    print('\nPlease install in the virtual environment:')
    print('  pip install colour-science numpy imageio Pillow matplotlib')
" 2>&1

echo.
echo [INFO] Starting main application: Aurhythm.py
echo ========================================
echo.

REM 运行主程序
python Aurhythm.py

REM 检查程序退出状态
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ========================================
    echo [ERROR] Program exited with error code: %ERRORLEVEL%
    echo.
    echo Troubleshooting steps:
    echo 1. Ensure virtual environment is activated correctly
    echo 2. Check all dependencies are installed: 
    echo      pip list
    echo 3. Run python directly to see error details:
    echo      python Aurhythm.py
    echo.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ========================================
echo [INFO] Program finished successfully
echo ========================================
pause
