@echo off
rem Double-click: all Case 0 runs (Ra 1e3..1e6) on all meshes (40, 80, 160) in Ansys Fluent.
rem Log: ansys\_fluent_runs\case0\run_log.txt
cd /d "%~dp0.."
".venv\Scripts\python.exe" -m ansys.run_case0
