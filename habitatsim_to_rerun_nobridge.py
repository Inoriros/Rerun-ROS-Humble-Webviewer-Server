#!/usr/bin/env python3
"""
habitatsim_to_rerun_nobridge.py

All 3D poses (cameras, base, occupancy map points, point clouds) are in the
ROS 'map' frame (MAP_FRAME), so everything aligns with the occupancy grid.
"""

import sys
import time
import traceback
import inspect

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

# Added PointStamped to imports
from sensor_msgs.msg import CompressedImage, Image, Imu, PointCloud2
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from geometry_msgs.msg import PointStamped
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker, MarkerArray

# Import DetectedObjectArray for target object visualization
try:
    from mask_msgs.msg import DetectedObjectArray
    HAS_MASK_MSGS = True
except ImportError:
    HAS_MASK_MSGS = False

import rerun as rr
import tf2_ros

try:
    from sensor_msgs_py import point_cloud2 as pc2
except ImportError:
    pc2 = None

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
LOG_WAYPOINT = True
LOG_PATH = True

# Flags for value map logging
LOG_VALUE_SCAN = True
LOG_EXPLORED_VALUE_MAP = False

# Flags for SuperMap semantic mapping visualization
LOG_SUPERMAP_OBJ_POINTS = False
LOG_SUPERMAP_OBJ_BOXES = False
LOG_SUPERMAP_ANNOTATED_IMAGES = False

# Flags for DualMap semantic mapping visualization
LOG_DUALMAP_LOCAL_SEMANTIC = True
LOG_DUALMAP_GLOBAL_SEMANTIC = True
LOG_DUALMAP_LOCAL_RGB = False  # RGB point clouds (optional, can be large)
LOG_DUALMAP_GLOBAL_RGB = False
LOG_DUALMAP_LOCAL_BBOXES = True  # Local map object bounding boxes
LOG_DUALMAP_GLOBAL_BBOXES = True  # Global map object bounding boxes
LOG_DUALMAP_ANNOTATED_IMAGES = True  # YOLO-World detection visualization (per camera)


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

# Colors: obstacle/occupied cells should use pure red; other visuals must avoid red.
# `OCCUPIED_COLOR` is the obstacle grid color (pure red). The rainbow colormap
# endpoint remains orange to avoid colliding with obstacle red in overlays.
OCCUPIED_COLOR = np.array([255, 0, 0], dtype=np.uint8)
RAINBOW_HIGH_RGB = np.array([255, 128, 0], dtype=np.uint8)
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
        return arr.astype(np.float32) / 1000.0

    elif enc.startswith("32F"):
        arr = np.frombuffer(buf, dtype=np.float32, count=h * w).reshape((h, w))
        if getattr(msg, "is_bigendian", 0) == 1:
            arr = arr.byteswap().newbyteorder()
        return arr

    else:
        arr = np.frombuffer(buf, dtype=np.float32, count=h * w).reshape((h, w))
        if getattr(msg, "is_bigendian", 0) == 1:
            arr = arr.byteswap().newbyteorder()
        return arr


def color_image_from_ros(msg: Image) -> np.ndarray:
    enc = (msg.encoding or "").lower()
    h, w = msg.height, msg.width
    buf = msg.data

    if enc in ("bgr8", "8uc3"):
        bgr = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 3))
        return bgr[..., ::-1]
    elif enc == "rgb8":
        return np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 3))
    elif enc == "bgra8":
        bgra = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
        bgr = bgra[..., :3]
        return bgr[..., ::-1]
    elif enc == "rgba8":
        rgba = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
        return rgba[..., :3]
    elif enc in ("mono8", "8uc1"):
        gray = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 1))
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


def pointcloud2_to_xyz_and_color_downsampled(logger, msg: PointCloud2, stride: int = 1, max_points=None):
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

        kept = 0
        for i, p in enumerate(pc2.read_points(msg, skip_nans=False, field_names=field_names)):
            if stride > 1 and (i % stride) != 0:
                continue

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
            elif has_rgb_packed or has_rgb_separate or has_intensity:
                colors.append((0, 0, 0))

            kept += 1
            if max_points is not None and kept >= max_points:
                break

        positions = np.asarray(positions, dtype=np.float32)
        if colors:
            if len(colors) < len(positions):
                colors.extend([(0, 0, 0)] * (len(positions) - len(colors)))
            colors_arr = np.asarray(colors, dtype=np.uint8)
        else:
            colors_arr = None

        return positions, colors_arr

    except Exception as e:
        logger.warn(f"Failed to convert PointCloud2 (downsampled): {e}")
        return np.zeros((0, 3), np.float32), None


def send_point_cloud(logger, entity_path: str, pts: np.ndarray, colors, radius=None):
    if pts.size == 0:
        return
    try:
        # Construct kwargs dynamically to handle older/newer Rerun versions or missing radius
        kwargs = {"positions": pts}
        if colors is not None:
            kwargs["colors"] = colors
        if radius is not None:
            kwargs["radii"] = radius
            
        rr.log(entity_path, rr.Points3D(**kwargs))
    except Exception as e:
        logger.warn(f"Failed to log point cloud for {entity_path}: {e}")


def rainbow_colormap(values: np.ndarray) -> np.ndarray:
    """
    Map scalar values in [0, 1] -> RGB uint8 using a smooth multi-stop gradient:
    0.0  -> blue    (0, 0, 255)
    0.25 -> cyan    (0, 255, 255)
    0.5  -> green   (0, 255, 0)
    0.75 -> yellow  (255, 255, 0)
    1.0  -> orange  (255, 128, 0)  -- avoids pure red reserved for obstacles
    
    Uses linear interpolation between stops for smooth transitions.
    """
    v = np.clip(np.asarray(values, dtype=np.float32).ravel(), 0.0, 1.0)
    
    # Define color stops: (value, R, G, B) normalized to [0, 1]
    stops = np.array([
        [0.00, 0.0,  0.0,  1.0],   # blue
        [0.25, 0.0,  1.0,  1.0],   # cyan
        [0.50, 0.0,  1.0,  0.0],   # green
        [0.75, 1.0,  1.0,  0.0],   # yellow
        [1.00, 1.0,  0.5,  0.0],   # orange
    ], dtype=np.float32)
    
    # Output arrays
    r = np.empty_like(v)
    g = np.empty_like(v)
    b = np.empty_like(v)
    
    # Interpolate between stops
    for idx in range(len(stops) - 1):
        t0, r0, g0, b0 = stops[idx]
        t1, r1, g1, b1 = stops[idx + 1]
        
        mask = (v >= t0) & (v <= t1)
        if not np.any(mask):
            continue
        
        # Local interpolation factor in [0, 1]
        t_local = (v[mask] - t0) / (t1 - t0 + 1e-9)
        
        r[mask] = r0 + (r1 - r0) * t_local
        g[mask] = g0 + (g1 - g0) * t_local
        b[mask] = b0 + (b1 - b0) * t_local
    
    colors = np.stack([r, g, b], axis=-1)
    return (np.clip(colors, 0.0, 1.0) * 255.0).astype(np.uint8)


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


class HabitatToRerun(Node):
    def __init__(self):
        super().__init__("habitatsim_to_rerun")

        self.declare_parameter("rerun_addr", "127.0.0.1")
        self.declare_parameter("rerun_port", 9876)
        host = self.get_parameter("rerun_addr").get_parameter_value().string_value
        port = int(self.get_parameter("rerun_port").get_parameter_value().integer_value)

        self.declare_parameter("odometryTopic", DEFAULT_ODOM_TOPIC)
        self.odom_topic = self.get_parameter("odometryTopic").get_parameter_value().string_value

        self.declare_parameter("fx", 128.0)
        self.declare_parameter("fy", 128.0)
        self.declare_parameter("cx", 128.0)
        self.declare_parameter("cy", 128.0)

        self.fx = float(self.get_parameter("fx").get_parameter_value().double_value)
        self.fy = float(self.get_parameter("fy").get_parameter_value().double_value)
        self.cx = float(self.get_parameter("cx").get_parameter_value().double_value)
        self.cy = float(self.get_parameter("cy").get_parameter_value().double_value)

        self.declare_parameter("rgb_fx", 360.0)
        self.declare_parameter("rgb_fy", 360.0)
        self.declare_parameter("rgb_cx", 360.0)
        self.declare_parameter("rgb_cy", 360.0)
        self.rgb_fx = float(self.get_parameter("rgb_fx").get_parameter_value().double_value)
        self.rgb_fy = float(self.get_parameter("rgb_fy").get_parameter_value().double_value)
        self.rgb_cx = float(self.get_parameter("rgb_cx").get_parameter_value().double_value)
        self.rgb_cy = float(self.get_parameter("rgb_cy").get_parameter_value().double_value)

        # Occupancy map Z value parameters
        self.declare_parameter("value_occupancy_map_z", 0.1)
        self.declare_parameter("value_occupancy_scan_z", 0.1)

        self.value_occupancy_map_z = float(
            self.get_parameter("value_occupancy_map_z").get_parameter_value().double_value
        )
        self.value_occupancy_scan_z = float(
            self.get_parameter("value_occupancy_scan_z").get_parameter_value().double_value
        )

        rr.init("habitatsim_rerun")
        ep = try_connect_grpc(self.get_logger(), host, port)
        self.get_logger().info(f"Rerun connected via {ep}")

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.current_odom = None
        self.has_odom = False

        self._occupancy_logged_once = False
        self.pinhole_logged = set()
        self.camera_depth_res = {}

        try:
            import cv2  # type: ignore
            self._cv2 = cv2
            self.get_logger().info("OpenCV (cv2) found; will resize RGB/detection to depth resolution.")
        except ImportError:
            self._cv2 = None
            self.get_logger().warn("OpenCV (cv2) not available; no resizing of RGB/detection.")

        self._terrain_pc_count = 0
        self._colored_pc_count = 0

        img_qos = qos_best_effort(10)

        self.camera_configs = {
            "frontleft": {
                "rgb_topic": "/spot/camera/frontleft/image/compressed",
                "depth_topic": "/spot/depth/frontleft/image",
                "detection_topic": "/proc_image/detection/image/frontleft",
                "dualmap_annotated_topic": "/dualmap/annotated_image/frontleft",
                "frame": "head_left_rgbd_optical",
            },
            "frontright": {
                "rgb_topic": "/spot/camera/frontright/image/compressed",
                "depth_topic": "/spot/depth/frontright/image",
                "detection_topic": "/proc_image/detection/image/frontright",
                "dualmap_annotated_topic": "/dualmap/annotated_image/frontright",
                "frame": "head_right_rgbd_optical",
            },
            "left": {
                "rgb_topic": "/spot/camera/left/image/compressed",
                "depth_topic": "/spot/depth/left/image",
                "detection_topic": "/proc_image/detection/image/left",
                "dualmap_annotated_topic": "/dualmap/annotated_image/left",
                "frame": "left_rgbd_optical",
            },
            "right": {
                "rgb_topic": "/spot/camera/right/image/compressed",
                "depth_topic": "/spot/depth/right/image",
                "detection_topic": "/proc_image/detection/image/right",
                "dualmap_annotated_topic": "/dualmap/annotated_image/right",
                "frame": "right_rgbd_optical",
            },
            "back": {
                "rgb_topic": "/spot/camera/back/image/compressed",
                "depth_topic": "/spot/depth/back/image",
                "detection_topic": "/proc_image/detection/image/back",
                "dualmap_annotated_topic": "/dualmap/annotated_image/back",
                "frame": "rear_rgbd_optical",
            },
        }

        for cam_name, cfg in self.camera_configs.items():
            base_entity = f"world/{cam_name}_camera_lowres"
            rgb_topic = cfg["rgb_topic"]
            depth_topic = cfg["depth_topic"]
            detection_topic = cfg["detection_topic"]
            dualmap_annotated_topic = cfg["dualmap_annotated_topic"]

            # RGB images (optional)
            if LOG_RGB_IMAGES:
                self.create_subscription(
                    CompressedImage,
                    rgb_topic,
                    lambda msg, base=base_entity: self.cb_rgb_camera_lowres(msg, base),
                    img_qos,
                )

            # Depth images (depth logging already controlled inside cb_depth via LOG_DEPTH_IMAGES)
            self.create_subscription(
                Image,
                depth_topic,
                lambda msg, base=base_entity, cam=cam_name: self.cb_depth_camera_lowres(msg, base, cam),
                img_qos,
            )

            # Detection images (optional)
            if LOG_DETECTION_IMAGES:
                self.create_subscription(
                    Image,
                    detection_topic,
                    lambda msg, base=base_entity: self.cb_detection_image(msg, base),
                    img_qos,
                )

            # DualMap annotated images (per camera)
            if LOG_DUALMAP_ANNOTATED_IMAGES:
                self.create_subscription(
                    Image,
                    dualmap_annotated_topic,
                    lambda msg, base=base_entity: self.cb_dualmap_annotated_image_percam(msg, base),
                    img_qos,
                )

        pc_qos = qos_best_effort(10)

        # Point cloud subscriptions
        if LOG_TERRAIN_MAP:
            self.create_subscription(
                PointCloud2,
                "/terrain_map_ext",
                lambda msg: self.cb_pointcloud(msg, "world/pointcloud/terrain_map_ext"),
                pc_qos,
            )

        if LOG_COLORED_REGISTERED_SCAN:
            self.create_subscription(
                PointCloud2,
                "/colored_registered_scan",
                lambda msg: self.cb_pointcloud(msg, "world/pointcloud/colored_registered_scan"),
                pc_qos,
            )
        
        # Value occupancy map subscriptions (OccupancyGrid → points, using cb_occupancy_grid)
        if LOG_VALUE_SCAN:
            self.create_subscription(
                OccupancyGrid,
                "/value_occupancy_scan",
                lambda msg: self.cb_value_occupancy_grid(
                    msg,
                    entity_base="world/value_occupancy_scan",
                    plane_z=self.value_occupancy_scan_z,
                ),
                qos_best_effort(1),
            )

        if LOG_EXPLORED_VALUE_MAP:
            self.create_subscription(
                OccupancyGrid,
                "/value_occupancy_map",
                lambda msg: self.cb_value_occupancy_grid(
                    msg,
                    entity_base="world/value_occupancy_map",
                    plane_z=self.value_occupancy_map_z,
                ),
                qos_best_effort(1),
            )
        

            
        # ----------------------------------------------------------------------
        # NEW TOPICS INTEGRATION
        # ----------------------------------------------------------------------
        if LOG_TARGETS:
            if HAS_MASK_MSGS:
                self.create_subscription(
                    DetectedObjectArray,
                    "/detected_objects",
                    # Green [0, 255, 0], Radius 0.08 (approx 2x standard grid cell visual)
                    lambda msg: self.cb_detected_objects(msg, "world/markers/detected_objects", [0, 255, 0], 0.08),
                    pc_qos,
                )
            else:
                self.get_logger().warn("mask_msgs not available; /detected_objects visualization disabled")
             
        if LOG_FRONTIERS:
             self.create_subscription(
                PointCloud2,
                "/frontier_points",
                # Purple [148, 0, 211], Radius 0.08 (approx 2x standard grid cell visual)
                lambda msg: self.cb_marker_pointcloud(msg, "world/markers/frontier_points", [148, 0, 211], 0.08),
                pc_qos,
            )
        
        # Selected goal point logging (white color, larger radius)
        if LOG_GOAL:
            self.create_subscription(
                PointStamped,
                "/selected_goal_point",
                lambda msg: self.cb_point_stamped(msg, "world/markers/selected_goal_point", [255, 255, 255], 0.15),
                qos_best_effort(10)
            )

        # Waypoint logging (yellow color, medium radius)
        if LOG_WAYPOINT:
            self.create_subscription(
                PointStamped,
                "/way_point",
                lambda msg: self.cb_point_stamped(msg, "world/markers/way_point", [255, 255, 0], 0.12),
                qos_best_effort(10)
            )

        # Path logging (yellow color)
        if LOG_PATH:
            self.create_subscription(
                Path,
                "/path",
                lambda msg: self.cb_path(msg, "world/markers/path", [255, 255, 0], 0.05),
                qos_best_effort(10)
            )
        # ----------------------------------------------------------------------

        if LOG_OCCUPANCY_GRID:
            self.create_subscription(
                OccupancyGrid,
                "/occupancy_grid",
                self.cb_occupancy_grid,
                qos_best_effort(1),
            )

        # ----------------------------------------------------------------------
        # SUPERMAP SEMANTIC MAPPING TOPICS
        # ----------------------------------------------------------------------
        # Object point cloud from SuperMap
        if LOG_SUPERMAP_OBJ_POINTS:
            self.create_subscription(
                PointCloud2,
                "/obj_points",
                lambda msg: self.cb_supermap_obj_points(msg, "world/supermap/obj_points"),
                pc_qos,
            )

        # Object bounding boxes from SuperMap
        if LOG_SUPERMAP_OBJ_BOXES:
            self.create_subscription(
                MarkerArray,
                "/obj_boxes",
                lambda msg: self.cb_supermap_obj_boxes(msg, "world/supermap/obj_boxes"),
                pc_qos,
            )

        # Annotated images from SuperMap (per camera)
        if LOG_SUPERMAP_ANNOTATED_IMAGES:
            supermap_camera_names = ["frontleft", "frontright", "left", "right", "back"]
            for cam_name in supermap_camera_names:
                self.create_subscription(
                    Image,
                    f"/annotated_image_{cam_name}",
                    lambda msg, cam=cam_name: self.cb_supermap_annotated_image(msg, f"world/{cam}_camera_lowres/supermap_detection"),
                    img_qos,
                )

        # ----------------------------------------------------------------------
        # DUALMAP SEMANTIC MAPPING TOPICS
        # ----------------------------------------------------------------------
        # Local semantic map (real-time, actively maintained)
        if LOG_DUALMAP_LOCAL_SEMANTIC:
            self.create_subscription(
                PointCloud2,
                "/local_map/semantic",
                lambda msg: self.cb_pointcloud(msg, "world/dualmap/local_semantic"),
                pc_qos,
            )

        # Global semantic map (accumulated over time)
        if LOG_DUALMAP_GLOBAL_SEMANTIC:
            self.create_subscription(
                PointCloud2,
                "/global_map/semantic",
                lambda msg: self.cb_pointcloud(msg, "world/dualmap/global_semantic"),
                pc_qos,
            )

        # RGB point clouds from DualMap (optional, can be large)
        if LOG_DUALMAP_LOCAL_RGB:
            self.create_subscription(
                PointCloud2,
                "/local_map/rgb",
                lambda msg: self.cb_pointcloud(msg, "world/dualmap/local_rgb"),
                pc_qos,
            )

        if LOG_DUALMAP_GLOBAL_RGB:
            self.create_subscription(
                PointCloud2,
                "/global_map/rgb",
                lambda msg: self.cb_pointcloud(msg, "world/dualmap/global_rgb"),
                pc_qos,
            )

        # DualMap bounding boxes
        if LOG_DUALMAP_LOCAL_BBOXES:
            self.create_subscription(
                MarkerArray,
                "/local_map/bboxes",
                lambda msg: self.cb_dualmap_bboxes(msg, "world/dualmap/local_bboxes"),
                10,
            )

        if LOG_DUALMAP_GLOBAL_BBOXES:
            self.create_subscription(
                MarkerArray,
                "/global_map/bboxes",
                lambda msg: self.cb_dualmap_bboxes(msg, "world/dualmap/global_bboxes"),
                10,
            )

        # Note: DualMap annotated images are now per-camera, subscribed in the camera_configs loop above
        # No centralized /annotated_image topic needed

        self.create_subscription(Imu, "/habitatsim/imu/data", self.cb_imu, 50)
        self.create_subscription(Odometry, self.odom_topic, self.cb_odom, 20)
        self.create_subscription(
            TFMessage, "/tf", lambda m: self.cb_tf(m, static=False), qos_tf(False)
        )
        self.create_subscription(
            TFMessage, "/tf_static", lambda m: self.cb_tf(m, static=True), qos_tf(True)
        )

        self.get_logger().info("Habitat → Rerun node with MAP-frame logging running.")

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
            rr.log(base_entity, rr.Pinhole(image_from_camera=K, resolution=[width, height]))
            self.pinhole_logged.add(base_entity)
        except TypeError:
            try:
                rr.log(base_entity, rr.Pinhole(resolution=[width, height]))
                self.pinhole_logged.add(base_entity)
            except Exception as e2:
                self.get_logger().warn(f"Failed to log Pinhole for {base_entity}: {e2}")
        except Exception as e:
            self.get_logger().warn(f"Failed to log Pinhole for {base_entity}: {e}")

    def cb_rgb_camera_lowres(self, msg: CompressedImage, base_entity: str):
        entity_path = f"{base_entity}/bgr"
        raw_data = bytes(msg.data)

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
                self.get_logger().warn(f"{entity_path}: unknown image format; assuming jpeg")

        target_res = self.camera_depth_res.get(base_entity, None)
        if target_res is None or self._cv2 is None:
            send_encoded_image(self.get_logger(), entity_path, raw_data, mime)
            return

        target_w, target_h = target_res

        try:
            np_data = np.frombuffer(raw_data, dtype=np.uint8)
            img_bgr = self._cv2.imdecode(np_data, self._cv2.IMREAD_COLOR)
            if img_bgr is None:
                send_encoded_image(self.get_logger(), entity_path, raw_data, mime)
                return

            resized = self._cv2.resize(img_bgr, (target_w, target_h), interpolation=self._cv2.INTER_AREA)
            success, buf = self._cv2.imencode(".jpg", resized)
            if not success:
                send_encoded_image(self.get_logger(), entity_path, raw_data, mime)
                return

            resized_bytes = buf.tobytes()
            send_encoded_image(self.get_logger(), entity_path, resized_bytes, "image/jpeg")
        except Exception as e:
            self.get_logger().warn(f"{entity_path}: RGB resize failed ({e}); logging original.")
            send_encoded_image(self.get_logger(), entity_path, raw_data, mime)

    def cb_detection_image(self, msg: Image, base_entity: str):
        entity_path = f"{base_entity}/detection"

        try:
            img_rgb = color_image_from_ros(msg)
        except Exception as e:
            self.get_logger().warn(f"{entity_path}: detection image convert failed: {e}")
            return

        target_res = self.camera_depth_res.get(base_entity, None)
        if target_res is not None and self._cv2 is not None:
            target_w, target_h = target_res
            try:
                img_rgb = self._cv2.resize(img_rgb, (target_w, target_h), interpolation=self._cv2.INTER_AREA)
            except Exception as e:
                self.get_logger().warn(f"{entity_path}: detection resize failed ({e}); using original.")

        try:
            rr.log(entity_path, rr.Image(img_rgb))
        except Exception as e:
            self.get_logger().warn(f"{entity_path}: rr.Image logging failed: {e}")

    def cb_depth_camera_lowres(self, msg: Image, base_entity: str, cam_name: str):
        entity_path = f"{base_entity}/depth"

        self.camera_depth_res[base_entity] = (msg.width, msg.height)
        self.maybe_log_pinhole_for_camera(base_entity, msg.width, msg.height)

        if LOG_DEPTH_IMAGES:
            try:
                depth_m = depth_array_from_ros(msg)
                send_depth_image(self.get_logger(), entity_path, depth_m)
            except Exception as e:
                self.get_logger().warn(f"Depth convert/log failed on {entity_path}: {e}")

        self.log_camera_pose_via_tf(base_entity, cam_name)

    def log_camera_pose_via_tf(self, base_entity: str, cam_name: str):
        cam_cfg = self.camera_configs.get(cam_name)
        if not cam_cfg:
            return

        camera_frame = cam_cfg.get("frame")
        if not camera_frame:
            return

        try:
            tfs = self.tf_buffer.lookup_transform(
                MAP_FRAME, camera_frame, rclpy.time.Time()
            )
        except Exception as e:
            self.get_logger().warn(
                f"TF lookup failed for camera {camera_frame} -> {MAP_FRAME}: {e}"
            )
            return

        T_map_cam = transform_to_matrix(
            (tfs.transform.translation.x,
             tfs.transform.translation.y,
             tfs.transform.translation.z),
            (tfs.transform.rotation.x,
             tfs.transform.rotation.y,
             tfs.transform.rotation.z,
             tfs.transform.rotation.w),
        )
        trans, quat_cam = matrix_to_transform(T_map_cam)
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
                rr.Transform3D(translation=translation, quaternion=quaternion),
            )
        except Exception as e:
            self.get_logger().warn(f"Failed to log camera pose at {base_entity}: {e}")

    # --- Pointcloud / IMU / occupancy / odom / TF ----------------------

    def cb_pointcloud(self, msg: PointCloud2, entity_path: str):
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
            pts, colors = pointcloud2_to_xyz_and_color_downsampled(self.get_logger(), msg, stride=1, max_points=None)
            pts = self._transform_points_to_map(msg.header.frame_id, pts)
            send_point_cloud(self.get_logger(), entity_path, pts, colors)
            return

        if counter_attr is not None:
            count = getattr(self, counter_attr, 0) + 1
            setattr(self, counter_attr, count)
            if count % every_n != 0:
                return

        pts, colors = pointcloud2_to_xyz_and_color_downsampled(
            self.get_logger(), msg, stride=stride, max_points=max_points
        )

        pts_map = self._transform_points_to_map(msg.header.frame_id, pts)

        self.get_logger().info(
            f"{entity_path}: logging {pts_map.shape[0]} points "
            f"(stride={stride}, max_points={max_points}, every_n={every_n})"
        )

        send_point_cloud(self.get_logger(), entity_path, pts_map, colors)
        
    def cb_marker_pointcloud(self, msg: PointCloud2, entity_path: str, color_rgb: list, radius: float):
        """
        Special callback for marker-like pointclouds (Targets, Frontiers)
        Allows overriding color and radius.
        """
        # Extract points (ignore internal color)
        pts, _ = pointcloud2_to_xyz_and_color_downsampled(self.get_logger(), msg, stride=1, max_points=None)
        
        # Transform to map frame
        pts_map = self._transform_points_to_map(msg.header.frame_id, pts)
        
        if pts_map.size == 0:
            return

        # Create uniform color array
        num_pts = pts_map.shape[0]
        colors = np.tile(color_rgb, (num_pts, 1)).astype(np.uint8)
        
        # Log with explicit radius
        send_point_cloud(self.get_logger(), entity_path, pts_map, colors, radius=radius)

    def cb_detected_objects(self, msg, entity_path: str, color_rgb: list, radius: float):
        """
        Callback for DetectedObjectArray messages from /detected_objects topic.
        Visualizes detected objects as colored points with labels.
        Each object has: name, position (x, y, z), and confidence.
        """
        if len(msg.objects) == 0:
            # Clear the visualization when no objects
            try:
                rr.log(entity_path, rr.Clear(recursive=False))
            except Exception:
                pass
            return

        # Extract positions from detected objects
        pts = np.array(
            [[obj.position.x, obj.position.y, obj.position.z] for obj in msg.objects],
            dtype=np.float32
        )

        # Transform to map frame if needed
        pts_map = self._transform_points_to_map(msg.header.frame_id, pts)

        if pts_map.size == 0:
            return

        # Create colors based on confidence (higher confidence = brighter green)
        num_pts = pts_map.shape[0]
        colors = np.zeros((num_pts, 3), dtype=np.uint8)
        for i, obj in enumerate(msg.objects):
            # Scale green intensity by confidence (min 100, max 255)
            intensity = int(100 + 155 * min(1.0, obj.confidence))
            colors[i] = [0, intensity, 0]  # Green with varying intensity

        # Log points
        try:
            rr.log(
                entity_path,
                rr.Points3D(
                    positions=pts_map,
                    colors=colors,
                    radii=radius
                )
            )
        except Exception as e:
            self.get_logger().warn(f"Failed to log detected objects at {entity_path}: {e}")

        # Log labels for each detection
        try:
            labels = [f"{obj.name}: {obj.confidence:.2f}" for obj in msg.objects]
            rr.log(
                f"{entity_path}/labels",
                rr.Points3D(
                    positions=pts_map,
                    labels=labels,
                    radii=radius * 0.5  # Smaller radius for label points
                )
            )
        except Exception as e:
            # Labels might not be supported in all Rerun versions
            pass

    def cb_point_stamped(self, msg: PointStamped, entity_path: str, color_rgb: list, radius: float):
        """
        Logs a single point with specified color and radius.
        - Goal point: white [255, 255, 255], radius 0.15
        - Waypoint: purple [148, 0, 211], radius 0.12
        """
        # Source point
        pt = np.array([[msg.point.x, msg.point.y, msg.point.z]], dtype=np.float32)
        
        # Transform to map
        pts_map = self._transform_points_to_map(msg.header.frame_id, pt)
        
        if pts_map.size == 0:
            return
            
        try:
            rr.log(
                entity_path, 
                rr.Points3D(
                    positions=pts_map, 
                    colors=color_rgb, 
                    radii=radius
                )
            )
        except Exception as e:
            self.get_logger().warn(f"Failed to log {entity_path}: {e}")

    def cb_path(self, msg: Path, entity_path: str, color_rgb: list, radius: float):
        """
        Logs a nav_msgs/Path as a line strip with specified color.
        """
        if len(msg.poses) == 0:
            return

        # Extract positions from all poses
        pts = np.array(
            [[pose.pose.position.x, pose.pose.position.y, pose.pose.position.z] for pose in msg.poses],
            dtype=np.float32
        )

        # Transform to map frame
        pts_map = self._transform_points_to_map(msg.header.frame_id, pts)

        if pts_map.size == 0:
            return

        # Log as LineStrips3D for path visualization
        try:
            rr.log(
                entity_path,
                rr.LineStrips3D(
                    strips=[pts_map],
                    colors=[color_rgb],
                    radii=radius
                )
            )
        except Exception as e:
            self.get_logger().warn(f"Failed to log path {entity_path}: {e}")

    def _transform_points_to_map(self, source_frame: str, pts: np.ndarray) -> np.ndarray:
        if pts.size == 0:
            return pts

        if not source_frame or source_frame == MAP_FRAME:
            return pts

        try:
            tfs = self.tf_buffer.lookup_transform(
                MAP_FRAME, source_frame, rclpy.time.Time()
            )
        except Exception as e:
            self.get_logger().warn(
                f"TF lookup failed for pointcloud {source_frame} -> {MAP_FRAME}: {e}"
            )
            return pts

        T_map_from_src = transform_to_matrix(
            (tfs.transform.translation.x,
             tfs.transform.translation.y,
             tfs.transform.translation.z),
            (tfs.transform.rotation.x,
             tfs.transform.rotation.y,
             tfs.transform.rotation.z,
             tfs.transform.rotation.w),
        )
        R = T_map_from_src[:3, :3]
        t = T_map_from_src[:3, 3]

        pts_map = (R @ pts.T).T + t
        return pts_map

    def cb_occupancy_grid(self, msg: OccupancyGrid, entity_base: str = "world/occupancy_grid", plane_z=None):
        if msg.info.width == 0 or msg.info.height == 0:
            return

        # Is this the main /occupancy_grid or one of the value overlays?
        is_main_grid = (entity_base == "world/occupancy_grid")

        if is_main_grid and LOG_OCCUPANCY_ONCE and self._occupancy_logged_once:
            return

        w = msg.info.width
        h = msg.info.height
        res = msg.info.resolution

        grid_frame = msg.header.frame_id or MAP_FRAME

        if grid_frame != MAP_FRAME:
            self.get_logger().warn(
                f"OccupancyGrid frame_id {grid_frame} != MAP_FRAME {MAP_FRAME}; "
                f"this code assumes they are the same."
            )

        data = np.array(msg.data, dtype=np.int16).reshape((h, w))

        # Grayscale image as before
        img = np.zeros((h, w), dtype=np.uint8)
        img[data < 0] = 128
        img[data == 0] = 0
        img[data > 0] = 255

        rgb = np.stack([img, img, img], axis=-1)

        origin = msg.info.origin
        p = origin.position
        q = origin.orientation

        # Decide world Z for this grid:
        # - None  → use origin z (main /occupancy_grid behaviour)
        # - float → explicit plane height (for value layers)
        if plane_z is None:
            plane_z_used = p.z
        else:
            plane_z_used = float(plane_z)

        # Log origin transform INCLUDING manual XY offset and custom Z
        try:
            rr.log(
                entity_base,
                rr.Transform3D(
                    translation=[p.x + OCC_OFFSET_X, p.y + OCC_OFFSET_Y, plane_z_used],
                    quaternion=[q.x, q.y, q.z, q.w],
                ),
                static=is_main_grid and LOG_OCCUPANCY_ONCE,
            )
        except Exception as e:
            self.get_logger().warn(f"Failed to log {entity_base} transform: {e}")

        # Log image at the same subtree
        try:
            rr.log(f"{entity_base}/image", rr.Image(rgb))
        except Exception as e:
            self.get_logger().warn(f"Failed to log {entity_base} image: {e}")

        occ_thresh = 50
        free_thresh = 10

        flat = data.reshape(-1)
        known_mask = flat >= 0
        if not np.any(known_mask):
            return

        js, is_ = np.indices((h, w))
        is_flat = is_.reshape(-1)
        js_flat = js.reshape(-1)

        is_known = is_flat[known_mask]
        js_known = js_flat[known_mask]
        vals_known = flat[known_mask]

        x_local = (is_known.astype(np.float32) + 0.5) * res
        y_local = (js_known.astype(np.float32) + 0.5) * res
        z_local = np.zeros_like(x_local)

        pts_local = np.stack([x_local, y_local, z_local], axis=-1)

        R_origin = quat_to_matrix(q.x, q.y, q.z, q.w)
        # Use plane_z_used as the world Z for this layer
        t_origin = np.array([p.x, p.y, plane_z_used], dtype=np.float32)
        pts_map = (R_origin @ pts_local.T).T + t_origin

        # Manual 2D offset alignment (same as before)
        pts_map[:, 0] += OCC_OFFSET_X
        pts_map[:, 1] += OCC_OFFSET_Y

        colors = np.zeros((pts_map.shape[0], 3), dtype=np.uint8)

        unknown_mask = vals_known < 0
        free_mask = (vals_known >= 0) & (vals_known <= free_thresh)
        occ_mask = vals_known > occ_thresh

        colors[unknown_mask] = np.array([128, 128, 128], dtype=np.uint8)
        colors[free_mask] = np.array([50, 50, 50], dtype=np.uint8)
        colors[occ_mask] = OCCUPIED_COLOR

        try:
            rr.log(
                f"{entity_base}/cells",
                rr.Points3D(positions=pts_map, colors=colors),
                static=is_main_grid and LOG_OCCUPANCY_ONCE,
            )
        except Exception as e:
            self.get_logger().warn(f"Failed to log {entity_base} cells as Points3D: {e}")

        if is_main_grid and LOG_OCCUPANCY_ONCE:
            self._occupancy_logged_once = True

    
    def cb_value_occupancy_grid(self, msg: OccupancyGrid, entity_base: str, plane_z: float):
        """
        For /value_occupancy_map and /value_occupancy_scan:
          - data = -1  : unknown  → no points
          - data in [0, 1] (and a bit above) → scalar value
          - data > 1   : treated as 1
        Rendered as a rainbow-colored layer slightly above the main occupancy grid.
        """
        if msg.info.width == 0 or msg.info.height == 0:
            return

        w = msg.info.width
        h = msg.info.height
        res = msg.info.resolution

        grid_frame = msg.header.frame_id or MAP_FRAME
        if grid_frame != MAP_FRAME:
            self.get_logger().warn(
                f"Value OccupancyGrid frame_id {grid_frame} != MAP_FRAME {MAP_FRAME}; "
                f"this code assumes they are the same."
            )

        # Values as float, shape (h, w)
        # OccupancyGrid data is typically int8 with values in [0, 100] or [-1 for unknown]
        # We need to normalize to [0, 1] for the colormap
        data = np.array(msg.data, dtype=np.float32).reshape((h, w))

        origin = msg.info.origin
        p = origin.position
        q = origin.orientation

        # --- Select known cells: everything >= 0.0 ---
        flat = data.reshape(-1)
        known_mask = flat >= 0.0    # -1 (unknown) is skipped
        if not np.any(known_mask):
            return

        js, is_ = np.indices((h, w))
        is_flat = is_.reshape(-1)
        js_flat = js.reshape(-1)

        is_known = is_flat[known_mask]
        js_known = js_flat[known_mask]
        vals_known = flat[known_mask]

        # Normalize values to [0, 1] range
        # If values are already in [0, 1], this won't change them much
        # If values are in [0, 100], this normalizes them properly
        val_min = vals_known.min()
        val_max = vals_known.max()
        if val_max > 1.0:
            # Data is likely in [0, 100] range, normalize
            vals_clamped = np.clip(vals_known / 100.0, 0.0, 1.0)
        elif val_max - val_min > 1e-6:
            # Data has some range, normalize to full [0, 1]
            vals_clamped = (vals_known - val_min) / (val_max - val_min)
        else:
            # All values are the same
            vals_clamped = np.clip(vals_known, 0.0, 1.0)

        # --- Compute 3D positions for the known cells ---
        x_local = (is_known.astype(np.float32) + 0.5) * res
        y_local = (js_known.astype(np.float32) + 0.5) * res
        z_local = np.zeros_like(x_local)

        pts_local = np.stack([x_local, y_local, z_local], axis=-1)

        R_origin = quat_to_matrix(q.x, q.y, q.z, q.w)
        # Put this layer at a configurable Z (plane_z)
        t_origin = np.array([p.x, p.y, plane_z], dtype=np.float32)
        pts_map = (R_origin @ pts_local.T).T + t_origin

        # Manual XY offset, same as /occupancy_grid
        pts_map[:, 0] += OCC_OFFSET_X
        pts_map[:, 1] += OCC_OFFSET_Y

        # --- Rainbow colors for the points ---
        colors_pts = rainbow_colormap(vals_clamped)

        try:
            rr.log(
                f"{entity_base}/cells",
                rr.Points3D(positions=pts_map, colors=colors_pts),
            )
        except Exception as e:
            self.get_logger().warn(f"Failed to log {entity_base} value cells as Points3D: {e}")

        # --- Optional: 2D rainbow image of the value field ---
        try:
            data_flat = data.reshape(-1)
            img_colors = np.zeros((data_flat.shape[0], 3), dtype=np.uint8)

            img_known_mask = data_flat >= 0.0
            if np.any(img_known_mask):
                img_vals = data_flat[img_known_mask]
                # Use same normalization as for 3D points
                img_val_min = img_vals.min()
                img_val_max = img_vals.max()
                if img_val_max > 1.0:
                    img_vals_norm = np.clip(img_vals / 100.0, 0.0, 1.0)
                elif img_val_max - img_val_min > 1e-6:
                    img_vals_norm = (img_vals - img_val_min) / (img_val_max - img_val_min)
                else:
                    img_vals_norm = np.clip(img_vals, 0.0, 1.0)
                img_colors[img_known_mask] = rainbow_colormap(img_vals_norm)

            img_rgb = img_colors.reshape(h, w, 3)

            rr.log(f"{entity_base}/image", rr.Image(img_rgb))
        except Exception as e:
            self.get_logger().warn(f"Failed to log {entity_base} image: {e}")

        # --- Transform for the layer (so Rerun knows where it lives in 3D) ---
        try:
            rr.log(
                entity_base,
                rr.Transform3D(
                    translation=[p.x + OCC_OFFSET_X, p.y + OCC_OFFSET_Y, plane_z],
                    quaternion=[q.x, q.y, q.z, q.w],
                ),
            )
        except Exception as e:
            self.get_logger().warn(f"Failed to log {entity_base} transform: {e}")

    # --- SuperMap Semantic Mapping Callbacks -----------------------------------

    def cb_supermap_obj_points(self, msg: PointCloud2, entity_path: str):
        """
        Callback for SuperMap /obj_points topic.
        Visualizes semantic object point clouds with their colors.
        """
        pts, colors = pointcloud2_to_xyz_and_color_downsampled(
            self.get_logger(), msg, stride=1, max_points=None
        )

        # Transform to map frame if needed
        pts_map = self._transform_points_to_map(msg.header.frame_id, pts)

        if pts_map.size == 0:
            return

        # Log the object point cloud
        try:
            kwargs = {"positions": pts_map}
            if colors is not None:
                kwargs["colors"] = colors
            kwargs["radii"] = 0.03  # Slightly larger points for visibility

            rr.log(entity_path, rr.Points3D(**kwargs))
        except Exception as e:
            self.get_logger().warn(f"Failed to log SuperMap obj_points at {entity_path}: {e}")

    def cb_supermap_obj_boxes(self, msg: MarkerArray, entity_path: str):
        """
        Callback for SuperMap /obj_boxes topic.
        Visualizes 3D bounding boxes for detected objects.
        Handles both CUBE markers and LINE_LIST markers (type 5) that define box edges.
        """
        if len(msg.markers) == 0:
            return

        # Collect all line strips from LINE_LIST markers
        all_line_strips = []
        all_line_colors = []
        
        # Collect label info: position and text for each marker
        label_positions = []
        label_texts = []
        label_colors = []

        # Also collect CUBE markers if any
        boxes_positions = []
        boxes_half_sizes = []
        boxes_colors = []
        boxes_labels = []

        for marker in msg.markers:
            if marker.action == Marker.DELETEALL:
                continue

            # Get color
            color = [
                int(marker.color.r * 255),
                int(marker.color.g * 255),
                int(marker.color.b * 255),
                int(marker.color.a * 255)
            ]

            # Parse label from namespace - format is like "{'chair': 1}"
            raw_ns = marker.ns if marker.ns else ""
            # Try to parse the dict-like string to get clean name
            try:
                # Remove outer braces and parse
                # Format: {'chair': 1} or {chair: 1}
                import ast
                parsed = ast.literal_eval(raw_ns)
                if isinstance(parsed, dict):
                    for obj_name, count in parsed.items():
                        label = f"{obj_name} #{count}"
                        break
                else:
                    label = raw_ns
            except:
                # Fallback: just use raw namespace
                label = raw_ns if raw_ns else f"obj_{marker.id}"

            if marker.type == Marker.LINE_LIST and len(marker.points) >= 2:
                # LINE_LIST: points are pairs defining line segments
                # Convert pairs of points into line segments
                points = marker.points
                
                # Calculate centroid of box for label placement
                all_x = [p.x for p in points]
                all_y = [p.y for p in points]
                all_z = [p.z for p in points]
                centroid = [
                    (min(all_x) + max(all_x)) / 2.0,
                    (min(all_y) + max(all_y)) / 2.0,
                    max(all_z) + 0.05  # Place label slightly above the top
                ]
                
                for i in range(0, len(points) - 1, 2):
                    p1 = points[i]
                    p2 = points[i + 1]
                    line_strip = np.array([
                        [p1.x, p1.y, p1.z],
                        [p2.x, p2.y, p2.z]
                    ], dtype=np.float32)
                    all_line_strips.append(line_strip)
                    all_line_colors.append(color)
                
                # Store label info
                label_positions.append(centroid)
                label_texts.append(label)
                label_colors.append(color)

            elif marker.type == Marker.CUBE:
                # Handle CUBE markers as before
                pos = [marker.pose.position.x, marker.pose.position.y, marker.pose.position.z]
                half_size = [marker.scale.x / 2.0, marker.scale.y / 2.0, marker.scale.z / 2.0]
                boxes_positions.append(pos)
                boxes_half_sizes.append(half_size)
                boxes_colors.append(color)
                boxes_labels.append(label)

        # Log LINE_LIST markers as LineStrips3D
        if len(all_line_strips) > 0:
            try:
                rr.log(
                    f"{entity_path}/lines",
                    rr.LineStrips3D(
                        strips=all_line_strips,
                        colors=all_line_colors,
                        radii=0.01,  # Line thickness
                    )
                )
            except Exception as e:
                self.get_logger().warn(f"Failed to log SuperMap obj_boxes lines at {entity_path}: {e}")

        # Log labels as Points3D with text labels
        if len(label_positions) > 0:
            try:
                rr.log(
                    f"{entity_path}/labels",
                    rr.Points3D(
                        positions=np.array(label_positions, dtype=np.float32),
                        colors=np.array(label_colors, dtype=np.uint8),
                        labels=label_texts,
                        radii=0.02,  # Small point for label anchor
                    )
                )
            except Exception as e:
                self.get_logger().warn(f"Failed to log SuperMap obj_boxes labels at {entity_path}: {e}")

        # Log CUBE markers as Boxes3D
        if len(boxes_positions) > 0:
            boxes_positions = np.array(boxes_positions, dtype=np.float32)
            boxes_half_sizes = np.array(boxes_half_sizes, dtype=np.float32)
            boxes_colors = np.array(boxes_colors, dtype=np.uint8)

            try:
                rr.log(
                    f"{entity_path}/cubes",
                    rr.Boxes3D(
                        centers=boxes_positions,
                        half_sizes=boxes_half_sizes,
                        colors=boxes_colors,
                        labels=boxes_labels,
                    )
                )
            except Exception as e:
                self.get_logger().warn(f"Failed to log SuperMap obj_boxes cubes at {entity_path}: {e}")

    def cb_dualmap_bboxes(self, msg: MarkerArray, entity_path: str):
        """
        Callback for DualMap /global_map/bboxes topic.
        Visualizes 3D bounding boxes for detected objects in the global map.
        """
        if len(msg.markers) == 0:
            return

        boxes_positions = []
        boxes_half_sizes = []
        boxes_colors = []
        boxes_labels = []

        for marker in msg.markers:
            if marker.action == Marker.DELETEALL or marker.type != Marker.CUBE:
                continue

            # Get position
            pos = [marker.pose.position.x, marker.pose.position.y, marker.pose.position.z]
            
            # Get dimensions (marker.scale is full size, need half size)
            half_size = [marker.scale.x / 2.0, marker.scale.y / 2.0, marker.scale.z / 2.0]
            
            # Get color
            color = [
                int(marker.color.r * 255),
                int(marker.color.g * 255),
                int(marker.color.b * 255),
                int(marker.color.a * 255)
            ]
            
            # Create label from namespace or ID
            label = marker.ns if marker.ns else f"obj_{marker.id}"
            
            boxes_positions.append(pos)
            boxes_half_sizes.append(half_size)
            boxes_colors.append(color)
            boxes_labels.append(label)

        # Log all boxes as Boxes3D
        if len(boxes_positions) > 0:
            boxes_positions = np.array(boxes_positions, dtype=np.float32)
            boxes_half_sizes = np.array(boxes_half_sizes, dtype=np.float32)
            boxes_colors = np.array(boxes_colors, dtype=np.uint8)

            try:
                rr.log(
                    entity_path,
                    rr.Boxes3D(
                        centers=boxes_positions,
                        half_sizes=boxes_half_sizes,
                        colors=boxes_colors,
                        labels=boxes_labels,
                    )
                )
            except Exception as e:
                self.get_logger().warn(f"Failed to log DualMap bboxes at {entity_path}: {e}")

    def cb_dualmap_annotated_image_percam(self, msg: Image, base_entity: str):
        """
        Callback for DualMap per-camera annotated images.
        Visualizes YOLO-World detection-annotated images from semantic mapping for each camera.
        Images are attached to the camera entity, similar to detection images.
        """
        entity_path = f"{base_entity}/dualmap_detection"
        
        try:
            img_rgb = color_image_from_ros(msg)
        except Exception as e:
            self.get_logger().warn(f"{entity_path}: DualMap annotated image convert failed: {e}")
            return

        # Resize to match depth resolution if available
        target_res = self.camera_depth_res.get(base_entity, None)
        if target_res is not None and self._cv2 is not None:
            target_w, target_h = target_res
            try:
                img_rgb = self._cv2.resize(img_rgb, (target_w, target_h), interpolation=self._cv2.INTER_AREA)
            except Exception as e:
                self.get_logger().warn(f"{entity_path}: DualMap annotated image resize failed ({e}); using original.")

        try:
            rr.log(entity_path, rr.Image(img_rgb))
        except Exception as e:
            self.get_logger().warn(f"{entity_path}: rr.Image logging failed: {e}")

    def cb_supermap_annotated_image(self, msg: Image, entity_path: str):
        """
        Callback for SuperMap /annotated_image_{cam_name} topics.
        Visualizes detection-annotated images from semantic mapping.
        """
        try:
            img_rgb = color_image_from_ros(msg)
        except Exception as e:
            self.get_logger().warn(f"{entity_path}: SuperMap annotated image convert failed: {e}")
            return

        # Extract base_entity from entity_path (e.g., "world/frontleft_camera_lowres/supermap_detection" -> "world/frontleft_camera_lowres")
        base_entity = "/".join(entity_path.split("/")[:-1])
        target_res = self.camera_depth_res.get(base_entity, None)
        if target_res is not None and self._cv2 is not None:
            target_w, target_h = target_res
            try:
                img_rgb = self._cv2.resize(img_rgb, (target_w, target_h), interpolation=self._cv2.INTER_AREA)
            except Exception as e:
                self.get_logger().warn(f"{entity_path}: SuperMap annotated image resize failed ({e}); using original.")

        try:
            rr.log(entity_path, rr.Image(img_rgb))
        except Exception as e:
            self.get_logger().warn(f"{entity_path}: rr.Image logging failed: {e}")


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
            tfs = self.tf_buffer.lookup_transform(
                MAP_FRAME, BASE_FRAME, rclpy.time.Time()
            )
            rr.log(
                "world/base",
                rr.Transform3D(
                    translation=[tfs.transform.translation.x,
                                 tfs.transform.translation.y,
                                 tfs.transform.translation.z],
                    quaternion=[tfs.transform.rotation.x,
                                tfs.transform.rotation.y,
                                tfs.transform.rotation.z,
                                tfs.transform.rotation.w],
                ),
            )
        except Exception as e:
            self.get_logger().warn(
                f"Failed to log base pose in MAP_FRAME={MAP_FRAME}: {e}"
            )

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