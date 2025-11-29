# Rerun-ROS Humble-Webviewer-Server
 This is a repo help you visulize the ros2 RGBD images, XYZI and XYZRGB pointclouds, pose of cameras topics and tfs in your headless server using rerun webviewer, ros2 bridge, and server service.


## 1.Install rerun and other libs
```
sudo apt update
sudo apt install -y python3 python3-pip python3-venv

# Python ≥3.10 recommended
pip3 install --upgrade rerun-sdk
pip uninstall numpy
# Install a known stable version for ROS 2 Humble
pip install numpy==1.24.4
# You should now have the `rerun` CLI in PATH
rerun --version
```


## 2. Run the web viwer and (optional: the keyboard `/cmd_vel` controler)
You can disable the the keyboard `/cmd_vel` controler in the script.

Run this script to start subscribing the target ros2 topics and logging to rerun webviewer.
```
sh start_rerun_web.sh
```

## 3. DIY the topic settings for the rerun webviewer
You can descide to disable some topics, downsample some point clouds and log the point clouds less frequently by editing these parameters in the `habitatsim_to_rerun_nobridge.py` file:
```
# ----------------------------------------------------------------------
# Flags
# ----------------------------------------------------------------------
# Flags for image logging
LOG_RGB_IMAGES = False
LOG_DETECTION_IMAGES = True
LOG_DEPTH_IMAGES = False

# Flags for point cloud logging
LOG_TERRAIN_MAP = False
LOG_COLORED_REGISTERED_SCAN = False

# Flags for occupancy grid logging
LOG_OCCUPANCY_GRID = True
LOG_OCCUPANCY_ONCE = False  # if map is static, can log once

# Flags for candidate goal points - frontiers points, target object points, selected goal point
LOG_FRONTIERS = True
LOG_TARGETS = True
LOG_GOAL = True

# Flags for value map logging
LOG_VALUE_MAP = True
LOG_EXPLORED_VALUE_MAP = False


TERRAIN_DOWNSAMPLE_STRIDE = 4
COLORED_DOWNSAMPLE_STRIDE = 4

TERRAIN_MAX_POINTS = 20000
COLORED_MAX_POINTS = 20000

TERRAIN_LOG_EVERY_N = 2
COLORED_LOG_EVERY_N = 2

# Frames
MAP_FRAME = "map"
BASE_FRAME = "spot/body"
DEFAULT_ODOM_TOPIC = "/spot/platform/odom"

# Manual spatial offset applied to the occupancy grid (in MAP_FRAME, meters).
OCC_OFFSET_X = 15.0
OCC_OFFSET_Y = 15.0
```
