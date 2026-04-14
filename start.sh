#!/bin/bash
alembic upgrade head
uvicorn api.main:app --host 0.0.0.0 --port 7860 &
sleep 3

python app/static.py