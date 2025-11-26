#!/bin/bash

# Script to restart pathFollower node with proper environment

export ROS_DOMAIN_ID=88

cd /root/SAIR_platform
source install/setup.bash

# Start pathFollower (it will use parameters from the launch file's defaults)
ros2 run local_planner pathFollower &

echo "pathFollower restarted"
