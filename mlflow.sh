#!/bin/sh
source env/bin/activate
mlflow ui --port 8080 --backend-store-uri sqlite:///mlruns.db
