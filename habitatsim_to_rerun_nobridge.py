#!/usr/bin/env python3
"""
habitatsim_to_rerun_nobridge.py

Logs 5 low-res cameras in a "camera_lowres" layout with proper 3D pose:

  world/<cam>_camera_lowres
    world/<cam>_camera_lowres/bgr
    world/<cam>_camera_lowres/depth
    world/<cam>_camera_lowres/detection

Camera poses are computed like DepthInterface.cpp:
  - TF:   "None/body" -> depth frame
  - Odom: /None/platform/odom
  - Compose to get camera pose in world/map.

Also logs IMU, odometry, TF, and point clouds.

Load-control flags (set here in Python):
  - LOG_DEPTH_IMAGES: whether to log depth images
  - LOG_TERRAIN_MAP: whether to subscribe/log /terrain_map_ext
  - LOG_COLORED_REGISTERED_SCAN: whether to subscribe/log /colored_registered_scan

Downsampling / throttling (set here in Python):
  - TERRAIN_DOWNSAMPLE_STRIDE: keep every N-th point from terrain map
  - COLORED_DOWNSAMPLE_STRIDE: keep every N-th point from colored cloud
  - TERRAIN_MAX_POINTS: cap terrain points per frame (0/None = no cap)
  - COLORED_MAX_POINTS: cap colored points per frame (0/None = no cap)
  - TERRAIN_LOG_EVERY_N: log only every N-th terrain frame (1 = every frame)
  - COLORED_LOG_EVERY_N: log only every N-th colored frame (1 = every frame)
"""

import sys
import time
import traceback
import inspect

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import CompressedImage, Image, Imu, PointCloud2
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage

import rerun as rr
import tf2_ros

try:
    from sensor_msgs_py import point_cloud2 as pc2
except ImportError:
    pc2 = None

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

BASE_FRAME = "None/body"
DEFAULT_ODOM_TOPIC = "/None/platform/odom"


# -----------------------------------------------------------------------------


def qos_best_effort(depth=10):
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
    )


def qos_tf(transient=False):
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.TRANSIENT_LOCAL if transient else DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=100,
    )


def depth_array_from_ros(msg: Image) -> np.ndarray:
    enc = (msg.encoding or "").upper()
    h, w = msg.height, msg.width
    buf = msg.data

    if enc.startswith("16U"):
        arr = np.frombuffer(buf, dtype=np.uint16, count=h * w).reshape((h, w))
        if getattr(msg, "is_bigendian", 0) == 1:
            arr = arr.byteswap().newbyteorder()
        # convert to meters
        return arr.astype(np.float32) / 1000.0

    elif enc.startswith("32F"):
        arr = np.frombuffer(buf, dtype=np.float32, count=h * w).reshape((h, w))
        if getattr(msg, "is_bigendian", 0) == 1:
            arr = arr.byteswap().newbyteorder()
        # already meters
        return arr

    else:
        # best-effort float32
        arr = np.frombuffer(buf, dtype=np.float32, count=h * w).reshape((h, w))
        if getattr(msg, "is_bigendian", 0) == 1:
            arr = arr.byteswap().newbyteorder()
        return arr


def color_image_from_ros(msg: Image) -> np.ndarray:
    """
    Convert a sensor_msgs/Image to an RGB uint8 numpy array (H, W, 3).

    The detector publishes 'bgr8', but Rerun expects RGB. This function
    normalizes everything to RGB so colors look correct in the viewer.
    """
    enc = (msg.encoding or "").lower()
    h, w = msg.height, msg.width
    buf = msg.data

    if enc in ("bgr8", "8uc3"):
        # BGR -> RGB
        bgr = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 3))
        return bgr[..., ::-1]
    elif enc == "rgb8":
        rgb = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 3))
        return rgb
    elif enc == "bgra8":
        bgra = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
        bgr = bgra[..., :3]
        return bgr[..., ::-1]  # BGR -> RGB
    elif enc == "rgba8":
        rgba = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
        rgb = rgba[..., :3]  # RGB already
        return rgb
    elif enc in ("mono8", "8uc1"):
        gray = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 1))
        # replicate to 3 channels, same value in R/G/B
        return np.repeat(gray, 3, axis=2)
    else:
        raise RuntimeError(f"Unsupported detection image encoding: '{msg.encoding}'")


def try_construct_and_log(logger, entity_path: str, candidates, positional_values=(), keyword_values=None):
    keyword_values = keyword_values or {}
    tried = []
    for name in candidates:
        obj = getattr(rr, name, None)
        if obj is None:
            tried.append(f"{name}(missing)")
            continue
        try:
            sig = None
            try:
                sig = inspect.signature(obj)
            except Exception:
                sig = None

            kwargs = {}
            if sig:
                for p in sig.parameters.values():
                    pn = p.name.lower()
                    for candidate_key, val in keyword_values.items():
                        if candidate_key.lower() in pn or pn in candidate_key.lower():
                            kwargs[p.name] = val
                            break

                    if pn in ("blob", "data", "bytes", "encoded", "contents", "image", "buffer") and positional_values:
                        if p.name not in kwargs:
                            kwargs[p.name] = positional_values[0]

                    if pn in ("media_type", "mime", "mime_type") and len(positional_values) >= 2:
                        if p.name not in kwargs:
                            kwargs[p.name] = positional_values[1]

            try:
                if kwargs:
                    instance = obj(**kwargs)
                else:
                    instance = obj(*positional_values)
                rr.log(entity_path, instance)
                return True
            except TypeError as e:
                tried.append(f"{name} TypeError: {e}")
            except Exception as e:
                tried.append(f"{name} error: {e}\n{traceback.format_exc()}")

            # fallback positional-only attempt
            try:
                instance = obj(*positional_values)
                rr.log(entity_path, instance)
                return True
            except Exception as e:
                tried.append(f"{name} positional error: {e}")
                continue
        except Exception as exc:
            tried.append(f"{name} inspect/other error: {exc}")
            continue

    logger.warn(f"All attempts failed for {entity_path}. Tried: {tried}")
    return False


def send_encoded_image(logger, entity_path, blob: bytes, mime: str):
    tried = []
    for cls_name in ("EncodedImage", "EncodedImageExt", "EncodedImageV1"):
        cls = getattr(rr, cls_name, None)
        if cls is None:
            tried.append(f"{cls_name}(missing)")
            continue
        try:
            inst = cls(contents=blob, media_type=mime)
            rr.log(entity_path, inst)
            return True
        except Exception as e:
            tried.append(f"{cls_name} contents+media_type: {e}")
        try:
            inst = cls(contents=blob)
            rr.log(entity_path, inst)
            return True
        except Exception as e:
            tried.append(f"{cls_name} contents only: {e}")
    try:
        if hasattr(rr, "log_encoded_image"):
            rr.log_encoded_image(entity_path, blob, mime)
            return True
    except Exception as e:
        tried.append(f"log_encoded_image: {e}")
    try:
        if hasattr(rr, "log_image_bytes"):
            rr.log_image_bytes(entity_path, blob, mime)
            return True
    except Exception as e:
        tried.append(f"log_image_bytes: {e}")
    logger.warn(f"EncodedImage attempts failed for {entity_path}. Tried: {tried}")
    return False


def send_depth_image(logger, entity_path, depth_m: np.ndarray):
    """Depth array is in meters. meter=1.0 means values are already meters."""
    try:
        rr.log(entity_path, rr.DepthImage(depth_m, meter=1.0))
        return True
    except Exception as e:
        logger.warn(f"DepthImage log failed for {entity_path}: {e}")
        return False


def send_vec3_or_fallback(logger, entity_path, xyz):
    for name in ("Vec3D", "Vec3", "Vec3f", "Vec3Array"):
        cls = getattr(rr, name, None)
        if cls is None:
            continue
        try:
            instance = cls(xyz)
            rr.log(entity_path, instance)
            return True
        except Exception:
            try:
                instance = cls(*xyz)
                rr.log(entity_path, instance)
                return True
            except Exception:
                continue
    try:
        rr.log(
            entity_path,
            rr.Transform3D(
                translation=[float(xyz[0]), float(xyz[1]), float(xyz[2])],
                quaternion=[0.0, 0.0, 0.0, 1.0],
            ),
        )
        return True
    except Exception as e:
        logger.warn(f"Vec3 fallback failed for {entity_path}: {e}")
        return False


def try_connect_grpc(logger, host: str, port: int, timeout_s: float = 5.0):
    endpoints = [
        f"rerun://{host}:{port}",
        f"rerun+http://{host}:{port}/proxy",
        f"rerun+https://{host}:{port}/proxy",
    ]
    last_exc = None
    start = time.time()
    for ep in endpoints:
        try:
            logger.info(f"Trying to connect to Rerun endpoint: {ep}")
            rr.connect_grpc(ep)
            logger.info(f"Connected to Rerun using endpoint: {ep}")
            return ep
        except Exception as e:
            logger.warn(f"Connect attempt failed for {ep}: {e}")
            last_exc = e
        if time.time() - start > timeout_s:
            break
    raise RuntimeError(f"Could not connect to Rerun gRPC at {host}:{port}: {last_exc}")


def pointcloud2_to_xyz_and_color(logger, msg: PointCloud2):
    if pc2 is None:
        logger.warn("sensor_msgs_py.point_cloud2 not available; cannot decode PointCloud2")
        return np.zeros((0, 3), np.float32), None

    try:
        field_names = [f.name for f in msg.fields]

        has_intensity = "intensity" in field_names
        has_rgb_packed = ("rgb" in field_names) or ("rgba" in field_names)
        has_rgb_separate = all(name in field_names for name in ("r", "g", "b"))

        positions = []
        colors = []

        for p in pc2.read_points(msg, skip_nans=False, field_names=field_names):
            p_dict = dict(zip(field_names, p))
            try:
                x = float(p_dict.get("x", 0.0))
                y = float(p_dict.get("y", 0.0))
                z = float(p_dict.get("z", 0.0))
            except Exception:
                continue
            if np.isnan(x) or np.isnan(y) or np.isnan(z):
                continue
            positions.append((x, y, z))

            color_tuple = None
            try:
                if has_rgb_packed:
                    key = "rgb" if "rgb" in p_dict else "rgba"
                    rgb_val = p_dict[key]
                    if isinstance(rgb_val, (float, np.floating)):
                        rgb_uint32 = np.frombuffer(
                            np.float32(rgb_val).tobytes(), dtype=np.uint32
                        )[0]
                    else:
                        rgb_uint32 = int(rgb_val)
                    r = (rgb_uint32 >> 16) & 0xFF
                    g = (rgb_uint32 >> 8) & 0xFF
                    b = rgb_uint32 & 0xFF
                    color_tuple = (r, g, b)
                elif has_rgb_separate:
                    r_val = float(p_dict.get("r", 0.0))
                    g_val = float(p_dict.get("g", 0.0))
                    b_val = float(p_dict.get("b", 0.0))
                    if np.isnan(r_val) or np.isnan(g_val) or np.isnan(b_val):
                        r_val, g_val, b_val = 0.0, 0.0, 0.0
                    r_i = max(0, min(255, int(r_val)))
                    g_i = max(0, min(255, int(g_val)))
                    b_i = max(0, min(255, int(b_val)))
                    color_tuple = (r_i, g_i, b_i)
                elif has_intensity:
                    intensity = float(p_dict["intensity"])
                    if np.isnan(intensity):
                        intensity = 0.0
                    intensity = max(0.0, min(1.0, intensity))
                    r = int(intensity * 255)
                    g = int((1.0 - intensity) * 255)
                    b = 0
                    color_tuple = (r, g, b)
            except Exception:
                color_tuple = None

            if color_tuple is not None:
                colors.append(color_tuple)
            else:
                if has_rgb_packed or has_rgb_separate or has_intensity:
                    colors.append((0, 0, 0))

        positions = np.asarray(positions, dtype=np.float32)
        if colors:
            if len(colors) < len(positions):
                colors.extend([(0, 0, 0)] * (len(positions) - len(colors)))
            colors_arr = np.asarray(colors, dtype=np.uint8)
        else:
            colors_arr = None
        return positions, colors_arr

    except Exception as e:
        logger.warn(f"Failed to convert PointCloud2: {e}")
        return np.zeros((0, 3), np.float32), None


def send_point_cloud(logger, entity_path: str, pts: np.ndarray, colors):
    if pts.size == 0:
        return
    try:
        if colors is None:
            rr.log(entity_path, rr.Points3D(positions=pts))
        else:
            rr.log(entity_path, rr.Points3D(positions=pts, colors=colors))
    except Exception as e:
        logger.warn(f"Failed to log point cloud for {entity_path}: {e}")


# --- small SE(3) helpers ----------------------------------------------------


def quat_to_matrix(x, y, z, w):
    n = x * x + y * y + z * z + w * w
    if n < 1e-8:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1.0 - s * (yy + zz),     s * (xy - wz),         s * (xz + wy)],
        [    s * (xy + wz),   1.0 - s * (xx + zz),       s * (yz - wx)],
        [    s * (xz - wy),       s * (yz + wx),     1.0 - s * (xx + yy)],
    ])


def matrix_to_quat(R):
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    tr = m00 + m11 + m22
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S
    q = np.array([qx, qy, qz, qw])
    q /= np.linalg.norm(q) + 1e-12
    return q


def transform_to_matrix(translation, quat):
    tx, ty, tz = translation
    qx, qy, qz, qw = quat
    R = quat_to_matrix(qx, qy, qz, qw)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return T


def matrix_to_transform(T):
    R = T[:3, :3]
    t = T[:3, 3]
    q = matrix_to_quat(R)
    return t, q


# -----------------------------------------------------------------------------


class HabitatToRerun(Node):
    def __init__(self):
        super().__init__("habitatsim_to_rerun")

        # Rerun connection params
        self.declare_parameter("rerun_addr", "127.0.0.1")
        self.declare_parameter("rerun_port", 9876)
        host = self.get_parameter("rerun_addr").get_parameter_value().string_value
        port = int(self.get_parameter("rerun_port").get_parameter_value().integer_value)

        # Odom topic
        self.declare_parameter("odometryTopic", DEFAULT_ODOM_TOPIC)
        self.odom_topic = self.get_parameter("odometryTopic").get_parameter_value().string_value

        # Depth intrinsics (same as DepthInterface C++)
        self.declare_parameter("fx", 128.0)
        self.declare_parameter("fy", 128.0)
        self.declare_parameter("cx", 128.0)
        self.declare_parameter("cy", 128.0)

        self.fx = float(self.get_parameter("fx").get_parameter_value().double_value)
        self.fy = float(self.get_parameter("fy").get_parameter_value().double_value)
        self.cx = float(self.get_parameter("cx").get_parameter_value().double_value)
        self.cy = float(self.get_parameter("cy").get_parameter_value().double_value)

        # RGB intrinsics (if you ever need them later)
        self.declare_parameter("rgb_fx", 360.0)
        self.declare_parameter("rgb_fy", 360.0)
        self.declare_parameter("rgb_cx", 360.0)
        self.declare_parameter("rgb_cy", 360.0)
        self.rgb_fx = float(self.get_parameter("rgb_fx").get_parameter_value().double_value)
        self.rgb_fy = float(self.get_parameter("rgb_fy").get_parameter_value().double_value)
        self.rgb_cx = float(self.get_parameter("rgb_cx").get_parameter_value().double_value)
        self.rgb_cy = float(self.get_parameter("rgb_cy").get_parameter_value().double_value)

        rr.init("habitatsim_rerun")
        ep = try_connect_grpc(self.get_logger(), host, port)
        self.get_logger().info(f"Rerun connected via {ep}")
        self.get_logger().info(
            f"LOG_DEPTH_IMAGES={LOG_DEPTH_IMAGES}, "
            f"LOG_TERRAIN_MAP={LOG_TERRAIN_MAP}, "
            f"LOG_COLORED_REGISTERED_SCAN={LOG_COLORED_REGISTERED_SCAN}, "
            f"TERRAIN_DOWNSAMPLE_STRIDE={TERRAIN_DOWNSAMPLE_STRIDE}, "
            f"COLORED_DOWNSAMPLE_STRIDE={COLORED_DOWNSAMPLE_STRIDE}, "
            f"TERRAIN_MAX_POINTS={TERRAIN_MAX_POINTS}, "
            f"COLORED_MAX_POINTS={COLORED_MAX_POINTS}, "
            f"TERRAIN_LOG_EVERY_N={TERRAIN_LOG_EVERY_N}, "
            f"COLORED_LOG_EVERY_N={COLORED_LOG_EVERY_N}"
        )

        # TF & odom
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.current_odom = None
        self.has_odom = False

        # Track which cameras already have a Pinhole logged
        self.pinhole_logged = set()

        # Track per-camera depth resolution (so we can resize RGB/detections to match)
        self.camera_depth_res = {}  # base_entity -> (width, height)

        # Optional OpenCV for RGB/detection resizing
        try:
            import cv2  # type: ignore
            self._cv2 = cv2
            self.get_logger().info("OpenCV (cv2) found; will resize RGB/detection to depth resolution.")
        except ImportError:
            self._cv2 = None
            self.get_logger().warn(
                "OpenCV (cv2) not available; RGB/detection will NOT be resized to depth resolution."
            )

        # Counters for per-stream throttling of point clouds
        self._terrain_pc_count = 0
        self._colored_pc_count = 0

        img_qos = qos_best_effort(10)

        # 5 cameras: paths under "world/..."
        self.camera_configs = {
            "frontleft": {
                "rgb_topic": "/None/camera/frontleft/image/compressed",
                "depth_topic": "/None/depth/frontleft/image",
                "detection_topic": "/proc_image/detection/image/frontleft",
            },
            "frontright": {
                "rgb_topic": "/None/camera/frontright/image/compressed",
                "depth_topic": "/None/depth/frontright/image",
                "detection_topic": "/proc_image/detection/image/frontright",
            },
            "left": {
                "rgb_topic": "/None/camera/left/image/compressed",
                "depth_topic": "/None/depth/left/image",
                "detection_topic": "/proc_image/detection/image/left",
            },
            "right": {
                "rgb_topic": "/None/camera/right/image/compressed",
                "depth_topic": "/None/depth/right/image",
                "detection_topic": "/proc_image/detection/image/right",
            },
            "back": {
                "rgb_topic": "/None/camera/back/image/compressed",
                "depth_topic": "/None/depth/back/image",
                "detection_topic": "/proc_image/detection/image/back",
            },
        }

        for cam_name, cfg in self.camera_configs.items():
            base_entity = f"world/{cam_name}_camera_lowres"
            rgb_topic = cfg["rgb_topic"]
            depth_topic = cfg["depth_topic"]
            detection_topic = cfg["detection_topic"]

            # RGB (compressed)
            self.create_subscription(
                CompressedImage,
                rgb_topic,
                lambda msg, base=base_entity: self.cb_rgb_camera_lowres(msg, base),
                img_qos,
            )
            self.get_logger().info(
                f"Subscribed RGB for {cam_name}: {rgb_topic} -> {base_entity}/bgr"
            )

            # Depth
            self.create_subscription(
                Image,
                depth_topic,
                lambda msg, base=base_entity: self.cb_depth_camera_lowres(msg, base),
                img_qos,
            )
            self.get_logger().info(
                f"Subscribed depth for {cam_name}: {depth_topic} -> {base_entity}/depth"
            )

            # Detection overlay (sensor_msgs/Image, BGR in ROS, converted to RGB for Rerun)
            self.create_subscription(
                Image,
                detection_topic,
                lambda msg, base=base_entity: self.cb_detection_image(msg, base),
                img_qos,
            )
            self.get_logger().info(
                f"Subscribed detection for {cam_name}: {detection_topic} -> {base_entity}/detection"
            )

        # Point clouds
        pc_qos = qos_best_effort(10)

        if LOG_TERRAIN_MAP:
            self.create_subscription(
                PointCloud2,
                "/terrain_map_ext",
                lambda msg: self.cb_pointcloud(msg, "world/pointcloud/terrain_map_ext"),
                pc_qos,
            )
            self.get_logger().info(
                "Subscribed to terrain map: /terrain_map_ext -> world/pointcloud/terrain_map_ext"
            )
        else:
            self.get_logger().info("Skipping subscription to /terrain_map_ext (LOG_TERRAIN_MAP=False)")

        if LOG_COLORED_REGISTERED_SCAN:
            self.create_subscription(
                PointCloud2,
                "/colored_registered_scan",
                lambda msg: self.cb_pointcloud(msg, "world/pointcloud/colored_registered_scan"),
                pc_qos,
            )
            self.get_logger().info(
                "Subscribed to colored point cloud: /colored_registered_scan -> world/pointcloud/colored_registered_scan"
            )
        else:
            self.get_logger().info("Skipping subscription to /colored_registered_scan (LOG_COLORED_REGISTERED_SCAN=False)")

        # IMU / Odom / TF
        self.create_subscription(Imu, "/habitatsim/imu/data", self.cb_imu, 50)
        self.create_subscription(Odometry, self.odom_topic, self.cb_odom, 20)
        self.create_subscription(
            TFMessage,
            "/tf",
            lambda m: self.cb_tf(m, static=False),
            qos_tf(False),
        )
        self.create_subscription(
            TFMessage,
            "/tf_static",
            lambda m: self.cb_tf(m, static=True),
            qos_tf(True),
        )

        self.get_logger().info("Habitat → Rerun node with camera_lowres Pinhole logging running.")

    # --- Cameras -------------------------------------------------------

    def maybe_log_pinhole_for_camera(self, base_entity: str, width: int, height: int):
        if base_entity in self.pinhole_logged:
            return

        K = np.array(
            [
                [self.fx, 0.0,       self.cx],
                [0.0,     self.fy,   self.cy],
                [0.0,     0.0,       1.0],
            ],
            dtype=float,
        )

        try:
            rr.log(
                base_entity,
                rr.Pinhole(
                    image_from_camera=K,
                    resolution=[width, height],
                ),
            )
            self.pinhole_logged.add(base_entity)
            self.get_logger().info(
                f"Logged Pinhole for {base_entity} (w={width}, h={height}, "
                f"fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy})"
            )
        except TypeError as e:
            self.get_logger().warn(
                f"Pinhole(image_from_camera=..., resolution=...) failed for "
                f"{base_entity} ({e}); trying resolution-only fallback."
            )
            try:
                rr.log(
                    base_entity,
                    rr.Pinhole(
                        resolution=[width, height],
                    ),
                )
                self.pinhole_logged.add(base_entity)
                self.get_logger().info(
                    f"Logged fallback Pinhole (resolution-only) for {base_entity}"
                )
            except Exception as e2:
                self.get_logger().warn(
                    f"Failed to log Pinhole for {base_entity}: {e2}"
                )
        except Exception as e:
            self.get_logger().warn(
                f"Failed to log Pinhole for {base_entity}: {e}"
            )

    def cb_rgb_camera_lowres(self, msg: CompressedImage, base_entity: str):
        """
        Log low-res RGB image at <base_entity>/bgr.

        If we know the matching depth resolution for this camera, and cv2 is
        available, we:
          - decode the JPEG/PNG
          - resize to the depth resolution
          - re-encode and log
        so that the RGB texture visually matches the depth/camera frustum size.
        """
        entity_path = f"{base_entity}/bgr"
        raw_data = bytes(msg.data)

        # Determine mime from magic bytes / format
        if raw_data.startswith(b"\xff\xd8\xff"):
            mime = "image/jpeg"
        elif raw_data.startswith(b"\x89PNG\r\n\x1a\n"):
            mime = "image/png"
        elif raw_data[:4] == b"RIFF" and raw_data[8:12] == b"WEBP":
            mime = "image/webp"
        else:
            fmt = (msg.format or "").lower()
            if "jpeg" in fmt or "jpg" in fmt:
                mime = "image/jpeg"
            elif "png" in fmt:
                mime = "image/png"
            else:
                mime = "image/jpeg"
                self.get_logger().warn(
                    f"{entity_path}: unknown compressed image format; assuming jpeg"
                )

        # If we don't know the depth resolution yet or have no cv2, just log original
        target_res = self.camera_depth_res.get(base_entity, None)
        if target_res is None or self._cv2 is None:
            ok = send_encoded_image(self.get_logger(), entity_path, raw_data, mime)
            if not ok:
                self.get_logger().warn(
                    f"{entity_path}: encoded image logging ultimately failed (skipped)"
                )
            return

        target_w, target_h = target_res

        try:
            np_data = np.frombuffer(raw_data, dtype=np.uint8)
            img_bgr = self._cv2.imdecode(np_data, self._cv2.IMREAD_COLOR)
            if img_bgr is None:
                self.get_logger().warn(
                    f"{entity_path}: cv2.imdecode failed; logging original image."
                )
                ok = send_encoded_image(self.get_logger(), entity_path, raw_data, mime)
                if not ok:
                    self.get_logger().warn(
                        f"{entity_path}: encoded image logging ultimately failed (skipped)"
                    )
                return

            resized = self._cv2.resize(
                img_bgr,
                (target_w, target_h),
                interpolation=self._cv2.INTER_AREA,
            )

            success, buf = self._cv2.imencode(".jpg", resized)
            if not success:
                self.get_logger().warn(
                    f"{entity_path}: cv2.imencode failed; logging original image."
                )
                ok = send_encoded_image(self.get_logger(), entity_path, raw_data, mime)
                if not ok:
                    self.get_logger().warn(
                        f"{entity_path}: encoded image logging ultimately failed (skipped)"
                    )
                return

            resized_bytes = buf.tobytes()
            ok = send_encoded_image(self.get_logger(), entity_path, resized_bytes, "image/jpeg")
            if not ok:
                self.get_logger().warn(
                    f"{entity_path}: resized encoded image logging ultimately failed (skipped)"
                )

        except Exception as e:
            self.get_logger().warn(
                f"{entity_path}: RGB resize failed ({e}); logging original image."
            )
            ok = send_encoded_image(self.get_logger(), entity_path, raw_data, mime)
            if not ok:
                self.get_logger().warn(
                    f"{entity_path}: encoded image logging ultimately failed (skipped)"
                )

    def cb_detection_image(self, msg: Image, base_entity: str):
        """
        Log detection result image at <base_entity>/detection.

        sensor_msgs/Image (BGR in ROS) -> RGB numpy -> (optional) resize to depth resolution -> rr.Image
        """
        entity_path = f"{base_entity}/detection"

        try:
            img_rgb = color_image_from_ros(msg)  # this returns RGB
        except Exception as e:
            self.get_logger().warn(
                f"{entity_path}: detection image convert failed: {e}\n{traceback.format_exc()}"
            )
            return

        target_res = self.camera_depth_res.get(base_entity, None)
        if target_res is not None and self._cv2 is not None:
            target_w, target_h = target_res
            try:
                # OpenCV treats this as "BGR", but we're only resizing, so channel order doesn't matter.
                resized = self._cv2.resize(
                    img_rgb,
                    (target_w, target_h),
                    interpolation=self._cv2.INTER_AREA,
                )
                img_to_log = resized
            except Exception as e:
                self.get_logger().warn(
                    f"{entity_path}: detection resize failed ({e}); logging original size."
                )
                img_to_log = img_rgb
        else:
            img_to_log = img_rgb

        try:
            rr.log(entity_path, rr.Image(img_to_log))
        except Exception as e:
            self.get_logger().warn(
                f"{entity_path}: rr.Image logging failed: {e}"
            )

    def cb_depth_camera_lowres(self, msg: Image, base_entity: str):
        entity_path = f"{base_entity}/depth"

        # Remember this camera's depth resolution
        self.camera_depth_res[base_entity] = (msg.width, msg.height)

        # 1) Pinhole (once), independent of whether we log depth frames
        self.maybe_log_pinhole_for_camera(base_entity, msg.width, msg.height)

        # 2) Depth image (optional, controlled by LOG_DEPTH_IMAGES)
        if LOG_DEPTH_IMAGES:
            try:
                depth_m = depth_array_from_ros(msg)
                ok = send_depth_image(self.get_logger(), entity_path, depth_m)
                if not ok:
                    self.get_logger().warn(
                        f"{entity_path}: depth logging ultimately failed (skipped)"
                    )
            except Exception as e:
                self.get_logger().warn(
                    f"Depth convert/log failed on {entity_path}: {e}\n{traceback.format_exc()}"
                )

        # 3) Pose (always logged)
        self.log_camera_pose_from_depth_msg(msg, base_entity)

    def log_camera_pose_from_depth_msg(self, depth_msg: Image, base_entity: str):
        """
        Compute world_from_camera (map_from_camera) using:
          - TF:   BASE_FRAME -> depth_msg.header.frame_id  (base_from_camera)
          - Odom: self.current_odom                       (map_from_base)
        and log rr.Transform3D at base_entity.
        """
        if not self.has_odom or self.current_odom is None:
            return

        camera_frame = depth_msg.header.frame_id
        try:
            tfs = self.tf_buffer.lookup_transform(
                BASE_FRAME,      # target (base)
                camera_frame,    # source (camera)
                rclpy.time.Time(),
            )
        except Exception:
            return

        pos = self.current_odom.pose.pose.position
        quat = self.current_odom.pose.pose.orientation
        T_map_base = transform_to_matrix(
            (pos.x, pos.y, pos.z),
            (quat.x, quat.y, quat.z, quat.w),
        )

        t = tfs.transform.translation
        q = tfs.transform.rotation
        T_base_camera = transform_to_matrix(
            (t.x, t.y, t.z),
            (q.x, q.y, q.z, q.w),
        )

        T_map_camera = T_map_base @ T_base_camera

        trans, quat_cam = matrix_to_transform(T_map_camera)
        translation = [float(trans[0]), float(trans[1]), float(trans[2])]
        quaternion = [
            float(quat_cam[0]),
            float(quat_cam[1]),
            float(quat_cam[2]),
            float(quat_cam[3]),
        ]

        try:
            rr.log(
                base_entity,
                rr.Transform3D(
                    translation=translation,
                    quaternion=quaternion,
                ),
            )
        except Exception as e:
            self.get_logger().warn(
                f"Failed to log camera pose at {base_entity}: {e}"
            )

    # --- Pointcloud / IMU / odom / TF ---------------------------------

    def cb_pointcloud(self, msg: PointCloud2, entity_path: str):
        """
        Handle both terrain and colored point clouds with:
          - configurable downsampling,
          - per-stream frame throttling (log every N-th frame),
          - max point caps per frame.
        """
        pts, colors = pointcloud2_to_xyz_and_color(self.get_logger(), msg)

        # Decide which config to use based on entity_path
        stride = 1
        max_points = None
        every_n = 1
        counter_attr = None

        if entity_path.endswith("terrain_map_ext"):
            stride = max(1, int(TERRAIN_DOWNSAMPLE_STRIDE))
            max_points = TERRAIN_MAX_POINTS if TERRAIN_MAX_POINTS and TERRAIN_MAX_POINTS > 0 else None
            every_n = max(1, int(TERRAIN_LOG_EVERY_N))
            counter_attr = "_terrain_pc_count"
        elif entity_path.endswith("colored_registered_scan"):
            stride = max(1, int(COLORED_DOWNSAMPLE_STRIDE))
            max_points = COLORED_MAX_POINTS if COLORED_MAX_POINTS and COLORED_MAX_POINTS > 0 else None
            every_n = max(1, int(COLORED_LOG_EVERY_N))
            counter_attr = "_colored_pc_count"
        else:
            # Unknown pointcloud path: no special throttling/downsampling
            send_point_cloud(self.get_logger(), entity_path, pts, colors)
            return

        # Per-stream frame throttling: log only every_n-th message
        if counter_attr is not None:
            count = getattr(self, counter_attr, 0) + 1
            setattr(self, counter_attr, count)
            if count % every_n != 0:
                # Skip this frame entirely
                return

        # Downsample points by stride
        if stride > 1 and pts.size > 0:
            pts = pts[::stride]
            if colors is not None:
                colors = colors[::stride]

        # Hard cap on max_points
        if max_points is not None and max_points > 0 and pts.shape[0] > max_points:
            pts = pts[:max_points]
            if colors is not None and colors.shape[0] > max_points:
                colors = colors[:max_points]

        send_point_cloud(self.get_logger(), entity_path, pts, colors)

    def cb_imu(self, msg: Imu):
        try:
            q = msg.orientation
            rr.log(
                "sensors/imu/orientation",
                rr.Transform3D(
                    translation=[0.0, 0.0, 0.0],
                    quaternion=[q.x, q.y, q.z, q.w],
                ),
            )
        except Exception as e:
            self.get_logger().warn(f"Failed to log IMU orientation: {e}")

        ang = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
        lin = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
        send_vec3_or_fallback(self.get_logger(), "sensors/imu/angular_velocity", ang)
        send_vec3_or_fallback(self.get_logger(), "sensors/imu/linear_accel", lin)

    def cb_odom(self, msg: Odometry):
        self.current_odom = msg
        self.has_odom = True
        try:
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            rr.log(
                "world/base",
                rr.Transform3D(
                    translation=[p.x, p.y, p.z],
                    quaternion=[q.x, q.y, q.z, q.w],
                ),
            )
        except Exception as e:
            self.get_logger().warn(f"Failed to log base pose from odom: {e}")

    def cb_tf(self, msg: TFMessage, static: bool):
        try:
            for t in msg.transforms:
                parent = t.header.frame_id.lstrip("/")
                child = t.child_frame_id.lstrip("/")
                tr = t.transform.translation
                ro = t.transform.rotation
                rr.log(
                    f"tf/{parent}/{child}",
                    rr.Transform3D(
                        translation=[tr.x, tr.y, tr.z],
                        quaternion=[ro.x, ro.y, ro.z, ro.w],
                    ),
                    static=static,
                )
        except Exception as e:
            self.get_logger().warn(
                f"Failed to log TF message: {e}\n{traceback.format_exc()}"
            )


def main():
    rclpy.init()
    node = None
    try:
        node = HabitatToRerun()
        rclpy.spin(node)
    except Exception as e:
        print(f"Fatal error running node: {e}", file=sys.stderr)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
