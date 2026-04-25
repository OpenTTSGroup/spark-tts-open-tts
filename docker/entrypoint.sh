#!/usr/bin/env bash
set -euo pipefail

# Engine defaults
: "${SPARKTTS_MODEL:=SparkAudio/Spark-TTS-0.5B}"
: "${SPARKTTS_DEVICE:=auto}"
: "${SPARKTTS_DTYPE:=bfloat16}"

# Service-level defaults
: "${VOICES_DIR:=/voices}"
: "${HOST:=0.0.0.0}"
: "${PORT:=8000}"
: "${LOG_LEVEL:=info}"
: "${CORS_ENABLED:=false}"
: "${PYTHONPATH:=/opt/api:/opt/api/engine}"
: "${SPARKTTS_ROOT:=/opt/api/engine}"

export SPARKTTS_MODEL SPARKTTS_DEVICE SPARKTTS_DTYPE \
       VOICES_DIR HOST PORT LOG_LEVEL CORS_ENABLED \
       PYTHONPATH SPARKTTS_ROOT

if [ "$#" -eq 0 ]; then
  exec uvicorn app.server:app --host "$HOST" --port "$PORT" --log-level "$LOG_LEVEL"
fi
exec "$@"
