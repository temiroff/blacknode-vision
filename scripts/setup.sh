#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS_DIR="$ROOT_DIR/ros2_ws"

source_setup() {
  local setup_file="$1"
  set +u
  # shellcheck source=/dev/null
  source "$setup_file"
  local status=$?
  set -u
  return "$status"
}

if [[ -z "${ROS_DISTRO:-}" || -z "$(command -v ros2 2>/dev/null || true)" ]]; then
  if [[ -f /opt/ros/jazzy/setup.bash ]]; then
    source_setup /opt/ros/jazzy/setup.bash
  elif [[ -n "${ROS_DISTRO:-}" && -f "/opt/ros/$ROS_DISTRO/setup.bash" ]]; then
    source_setup "/opt/ros/$ROS_DISTRO/setup.bash"
  else
    echo "ERROR: ROS 2 was not found. Install ROS 2 Jazzy or source your ROS 2 setup before package setup." >&2
    exit 1
  fi
fi

if ! command -v colcon >/dev/null 2>&1; then
  echo "ERROR: colcon is required. On Ubuntu: sudo apt install python3-colcon-common-extensions" >&2
  exit 1
fi

cd "$WS_DIR"
colcon build --symlink-install

echo
echo "Built Blacknode Vision ROS workspace:"
echo "  $WS_DIR/install/setup.bash"
echo
echo "Start Blacknode with ./start.sh; it will source this package workspace automatically."
