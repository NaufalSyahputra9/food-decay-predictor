#!/bin/bash
alembic upgrade head
uvicorn api.main:app --host 127.0.0.1 --port 8000 &

sleep 3

python app/static.py