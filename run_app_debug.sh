#!/bin/sh
./create_env.sh
source env/bin/activate
uvicorn asgi:app --reload --host 0.0.0.0 --port 5000 --proxy-headers
