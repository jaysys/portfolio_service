#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${APP_ENV:-}" ]]; then
  MODE_INPUT="${APP_ENV}"
elif [[ -f "${ROOT_DIR}/.env.dev" ]]; then
  MODE_INPUT="development"
elif [[ -f "${ROOT_DIR}/.env.prod" ]]; then
  MODE_INPUT="production"
else
  MODE_INPUT="development"
fi

stop_by_pid_file() {
  local pid_file="$1"
  local label="$2"

  if [[ ! -f "${pid_file}" ]]; then
    echo "[${label}] pid file not found. service may already be stopped."
    return 0
  fi

  local pid
  pid="$(cat "${pid_file}" || true)"
  if [[ -z "${pid}" ]]; then
    echo "[${label}] invalid pid file. removing."
    rm -f "${pid_file}"
    return 0
  fi

  if ! kill -0 "${pid}" 2>/dev/null; then
    echo "[${label}] process ${pid} not running. cleaning pid file."
    rm -f "${pid_file}"
    return 0
  fi

  kill "${pid}" 2>/dev/null || true
  for _ in {1..20}; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      rm -f "${pid_file}"
      echo "[${label}] stopped (pid: ${pid})"
      return 0
    fi
    sleep 0.2
  done

  echo "[${label}] graceful stop timed out. sending SIGKILL to pid ${pid}"
  kill -9 "${pid}" 2>/dev/null || true
  rm -f "${pid_file}"
  echo "[${label}] stopped (forced)"
}

case "${MODE_INPUT}" in
  prod|production)
    stop_by_pid_file "${ROOT_DIR}/.portfolio_service.prod.pid" "production"
    ;;
  *)
    stop_by_pid_file "${ROOT_DIR}/.portfolio_service.dev.pid" "development"
    ;;
esac
