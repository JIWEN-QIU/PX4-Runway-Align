#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PX4_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BUILD_DIR="${PX4_DIR}/build/px4_sitl_default"
ROS_SETUP="/opt/ros/noetic/setup.bash"
CATKIN_SETUP="${HOME}/catkin_ws/devel/setup.bash"

if [ ! -f "${ROS_SETUP}" ]; then
	echo "ROS setup not found: ${ROS_SETUP}"
	exit 1
fi

cd "${PX4_DIR}"

source "${ROS_SETUP}"

if [ -f "${CATKIN_SETUP}" ]; then
	source "${CATKIN_SETUP}"
fi

export GAZEBO_PLUGIN_PATH="${GAZEBO_PLUGIN_PATH:-}"
export GAZEBO_MODEL_PATH="${GAZEBO_MODEL_PATH:-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

source "${PX4_DIR}/Tools/simulation/gazebo-classic/setup_gazebo.bash" "${PX4_DIR}" "${BUILD_DIR}"

export ROS_PACKAGE_PATH="${ROS_PACKAGE_PATH:-}:${PX4_DIR}:${PX4_DIR}/Tools/simulation/gazebo-classic/sitl_gazebo-classic"
export PATH="${BUILD_DIR}/bin:${PATH}"

roslaunch px4 runway_edge_detector.launch "$@"
