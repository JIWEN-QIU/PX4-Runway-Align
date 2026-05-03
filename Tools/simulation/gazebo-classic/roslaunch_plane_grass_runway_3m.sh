#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PX4_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BUILD_DIR="${PX4_DIR}/build/px4_sitl_default"
ROS_SETUP="/opt/ros/noetic/setup.bash"
CATKIN_SETUP="${HOME}/catkin_ws/devel/setup.bash"
WORLD_PATH="${PX4_DIR}/Tools/simulation/gazebo-classic/sitl_gazebo-classic/worlds/grass_runway_3m.world"
SDF_PATH="${PX4_DIR}/Tools/simulation/gazebo-classic/sitl_gazebo-classic/models/plane/plane.sdf"

if [ ! -f "${ROS_SETUP}" ]; then
	echo "ROS setup not found: ${ROS_SETUP}"
	exit 1
fi

if [ ! -x "${BUILD_DIR}/bin/px4" ]; then
	echo "PX4 SITL binary not found: ${BUILD_DIR}/bin/px4"
	echo "Run: make px4_sitl_default gazebo-classic"
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

# Clean up stale PX4 / Gazebo processes from previous failed launches.
pkill -9 -f gzserver >/dev/null 2>&1 || true
pkill -9 -f gzclient >/dev/null 2>&1 || true
pkill -9 -x px4 >/dev/null 2>&1 || true

roslaunch px4 mavros_posix_sitl.launch \
	vehicle:=plane \
	world:="${WORLD_PATH}" \
	sdf:="${SDF_PATH}" \
	x:=0 y:=0 z:=0.2 R:=0 P:=0 Y:=0 \
	gui:=true
