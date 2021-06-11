#!/bin/bash
# Load backups (wait 20 sec for ODES node initialization)
sleep 20 && python3 api/load_backups.py

# Launch application
gunicorn api.application:app -b 0.0.0.0 -k uvicorn.workers.UvicornWorker --workers 1 --timeout 180