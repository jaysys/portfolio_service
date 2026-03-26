#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UVICORN_BIN="${ROOT_DIR}/venv/bin/uvicorn"
PORT="${PORT:-7300}"

if [[ -n "${APP_ENV:-}" ]]; then
  MODE_INPUT="${APP_ENV}"
elif [[ -f "${ROOT_DIR}/.env.dev" ]]; then
  MODE_INPUT="development"
elif [[ -f "${ROOT_DIR}/.env.prod" ]]; then
  MODE_INPUT="production"
else
  MODE_INPUT="development"
fi

case "${MODE_INPUT}" in
  prod|production)
    APP_ENV_VALUE="production"
    PID_FILE="${ROOT_DIR}/.portfolio_service.prod.pid"
    LOG_FILE="${ROOT_DIR}/app.prod.log"
    ;;
  *)
    APP_ENV_VALUE="development"
    PID_FILE="${ROOT_DIR}/.portfolio_service.dev.pid"
    LOG_FILE="${ROOT_DIR}/app.dev.log"
    ;;
esac

if [[ ! -x "${UVICORN_BIN}" ]]; then
  echo "uvicorn not found: ${UVICORN_BIN}"
  echo "Create venv and install dependencies first."
  exit 1
fi

if [[ -f "${PID_FILE}" ]]; then
  EXISTING_PID="$(cat "${PID_FILE}" || true)"
  if [[ -n "${EXISTING_PID}" ]] && kill -0 "${EXISTING_PID}" 2>/dev/null; then
    echo "portfolio_service is already running (pid: ${EXISTING_PID})"
    exit 0
  fi
  rm -f "${PID_FILE}"
fi

cd "${ROOT_DIR}"

nohup env APP_ENV="${APP_ENV_VALUE}" "${UVICORN_BIN}" app:app --host 127.0.0.1 --port "${PORT}" >>"${LOG_FILE}" 2>&1 &
NEW_PID=$!
echo "${NEW_PID}" > "${PID_FILE}"

sleep 1
if kill -0 "${NEW_PID}" 2>/dev/null; then
  echo "portfolio_service started (${APP_ENV_VALUE}, pid: ${NEW_PID}, port: ${PORT})"
  echo "log: ${LOG_FILE}"
  exit 0
fi

echo "failed to start portfolio_service (${APP_ENV_VALUE}). check log: ${LOG_FILE}"
rm -f "${PID_FILE}"
exit 1
