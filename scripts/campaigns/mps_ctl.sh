#!/usr/bin/env bash
# Start/stop a per-user CUDA MPS daemon. Pipe+log dirs live on /CNSdata.
# Usage: mps_ctl.sh start | stop | status
set -u
export CUDA_MPS_PIPE_DIRECTORY=/CNSdata/mpaolett/mps_pipe
export CUDA_MPS_LOG_DIRECTORY=/CNSdata/mpaolett/mps_log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

case "${1:-status}" in
  start)
    if pgrep -u "$(id -u)" -f nvidia-cuda-mps-control >/dev/null 2>&1; then
      echo "MPS control daemon already running"
    else
      nvidia-cuda-mps-control -d && echo "MPS control daemon started"
    fi
    sleep 2
    echo get_server_list | nvidia-cuda-mps-control 2>&1 || true
    ;;
  stop)
    echo quit | nvidia-cuda-mps-control 2>&1 || echo "no daemon to stop"
    sleep 1
    pgrep -u "$(id -u)" -af "nvidia-cuda-mps" || echo "MPS fully stopped"
    ;;
  status)
    pgrep -u "$(id -u)" -af "nvidia-cuda-mps" || echo "no MPS processes"
    echo get_server_list | nvidia-cuda-mps-control 2>&1 || true
    ;;
esac
