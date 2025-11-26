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
# Simple Python flags you can flip:
# ----------------------------------------------------------------------
LOG_DEPTH_IMAGES = False               # log per-camera depth images
LOG_TERRAIN_MAP = True                 # subscribe/log /terrain_map_ext
LOG_COLORED_REGISTERED_SCAN = True     # subscribe/log /colored_registered_scan

# Downsampling (stride >= 1; 1 = no downsampling)
TERRAIN_DOWNSAMPLE_STRIDE = 4          # keep every N-th point from terrain
COLORED_DOWNSAMPLE_STRIDE = 4          # keep every N-th point from colored cloud

# Hard caps on number of points per frame (None or 0 = no cap)
TERRAIN_MAX_POINTS = 20000
COLORED_MAX_POINTS = 20000

# Log only every N-th pointcloud message (per stream). 1 = log every frame.
TERRAIN_LOG_EVERY_N = 2
COLORED_LOG_EVERY_N = 2
# ----------------------------------------------------------------------
```
