@echo off
rem Double-click: one Case 0 pilot run (Ra = 1e5, 40 x 40 mesh) in Ansys Fluent.
rem Log: ansys\_fluent_runs\case0\run_log.txt
cd /d "%~dp0.."
".venv\Scripts\python.exe" -m ansys.run_case0 --runs Ra1e5 --meshes coarse
