#!/bin/bash
source .venv/bin/activate
export LD_LIBRARY_PATH=$(python3 -c 'import nvidia.cublas.lib, nvidia.cudnn.lib; print(nvidia.cublas.lib.__path__[0] + ":" + nvidia.cudnn.lib.__path__[0])')
uvicorn server:app --host 0.0.0.0 --reload