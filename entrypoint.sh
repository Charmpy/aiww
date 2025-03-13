#!/bin/bash

set -e

source /opt/ros/jazzy/setup.bash
export GZ_SIM_RESOURCE_PATH=/aiww/src/gz_main/models:$GZ_SIM_RESOURCE_PATH

echo "Provided arguments: $@"

exec $@
