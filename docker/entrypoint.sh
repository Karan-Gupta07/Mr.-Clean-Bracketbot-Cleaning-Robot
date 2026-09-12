#!/bin/bash
set -e
source /opt/ros/jazzy/setup.bash
source "$RLBOT_REPO/ros2_ws/install/local_setup.bash"
exec "$@"
