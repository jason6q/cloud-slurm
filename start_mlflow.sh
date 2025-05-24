#!/bin/bash
echo "Attempting to start MLFlow server..."
mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri sqlite:///mlruns.db \
    --default-artifact-root /mnt/shared-slurm/mlruns