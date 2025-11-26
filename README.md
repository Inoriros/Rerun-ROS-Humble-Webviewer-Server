# Rerun-ROS Humble-Webviewer-erver
 This is a repo help you visulize the ros2 RGBD images, XYZI and XYZRGB pointclouds, pose of cameras topics and tfs in your headless server using rerun webviewer, ros2 bridge, and server service.


## Install rerun and other libs
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

