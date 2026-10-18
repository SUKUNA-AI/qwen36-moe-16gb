@echo off
if "%~1"=="key" (
  powershell.exe -NoProfile -File "C:\path\to\LLM\qwen36-api\get-api-key.ps1" "%~2" "%~3"
) else if "%~1"=="" (
  powershell.exe -NoProfile -File "C:\path\to\LLM\qwen36-api\manage.ps1" start
) else if "%~2"=="--profile" (
  powershell.exe -NoProfile -File "C:\path\to\LLM\qwen36-api\manage.ps1" -Action "%~1" -Profile "%~3"
) else (
  powershell.exe -NoProfile -File "C:\path\to\LLM\qwen36-api\manage.ps1" %*
)
