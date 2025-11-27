#!/usr/bin/env python3
import sys
import os
import termios
import tty
import threading
import time
import argparse
import subprocess
import signal

# # Set ROS_DOMAIN_ID
# os.environ['ROS_DOMAIN_ID'] = '88'

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

# Base speeds (scaled by command-line args)
BASE_LIN_VEL_FWD = 0.5   # m/s
BASE_LIN_VEL_LAT = 0.5   # m/s
BASE_ANG_VEL_Z   = 1.0   # rad/s

PUBLISH_RATE = 20.0
STOP_TIMEOUT = 0.2  # auto stop timeout


HELP = """
Keyboard teleop → /spot/cmd_vel (geometry_msgs/msg/Twist)

Controls:
    w : forward   (+x)
    s : backward  (-x)
    a : left      (+y)
    d : right     (-y)
    q : rotate CCW  (+z)
    e : rotate CW   (-z)
    x : stop
    p : PAUSE / UNPAUSE
    z or Ctrl+C : quit

No key for %.1fs → automatic stop.
"""


def getch():
    """Read a single character from stdin (blocking)."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def kill_path_follower():
    """Kill the pathFollower node."""
    try:
        subprocess.run(['pkill', '-9', '-f', 'pathFollower'], 
                      check=False, capture_output=True)
        print("Stopped pathFollower node")
        time.sleep(0.5)  # Give it time to stop
    except Exception as e:
        print(f"Warning: Could not stop pathFollower: {e}")


def restart_path_follower():
    """Restart the pathFollower node using external script."""
    try:
        print("Restarting pathFollower node...")
        script_path = os.path.join(os.path.dirname(__file__), 'restart_pathfollower.sh')
        subprocess.Popen(
            ['bash', script_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        time.sleep(1.5)  # Give it time to start
        print("pathFollower restarted")
    except Exception as e:
        print(f"Warning: Could not restart pathFollower: {e}")
        print("You may need to restart it manually")



class KeyboardCmdVel(Node):
    def __init__(self, move_scale, rotate_scale):
        super().__init__("keyboard_cmd_vel")
        self.pub = self.create_publisher(Twist, "/spot/cmd_vel", 10)

        self.move_scale = move_scale
        self.rotate_scale = rotate_scale

        self.current_twist = Twist()
        self.last_key_time = time.time()
        self.running = True
        self.paused = False

        print("Keyboard cmd_vel node started.")
        print(f"Move scale:   {self.move_scale}")
        print(f"Rotate scale: {self.rotate_scale}")
        print(HELP % STOP_TIMEOUT)

        self.timer = self.create_timer(1.0 / PUBLISH_RATE, self.publish_twist)
        self.thread = threading.Thread(target=self.keyboard_loop, daemon=True)
        self.thread.start()

    def keyboard_loop(self):
        while self.running:
            c = getch()
            now = time.time()

            # Quit
            if c in ("\x03", "z"):
                print("Quit requested.")
                self.running = False
                break

            # Pause toggle
            if c == "p":
                self.paused = not self.paused
                print("=== PAUSED ===" if self.paused else "=== UNPAUSED ===")
                self.current_twist = Twist()
                self.last_key_time = now
                continue

            if self.paused:
                continue

            twist = Twist()

            # Linear motions (scaled)
            if c == "w":
                twist.linear.x = BASE_LIN_VEL_FWD * self.move_scale
            elif c == "s":
                twist.linear.x = -BASE_LIN_VEL_FWD * self.move_scale
            elif c == "a":
                twist.linear.y = BASE_LIN_VEL_LAT * self.move_scale
            elif c == "d":
                twist.linear.y = -BASE_LIN_VEL_LAT * self.move_scale

            # Angular motions (scaled)
            elif c == "q":
                twist.angular.z = BASE_ANG_VEL_Z * self.rotate_scale
            elif c == "e":
                twist.angular.z = -BASE_ANG_VEL_Z * self.rotate_scale

            elif c == "x":
                twist = Twist()

            else:
                continue  # ignore unknown keys

            self.current_twist = twist
            self.last_key_time = now

    def publish_twist(self):
        """Publish twist (zero when paused or timed out)."""
        if self.paused:
            self.pub.publish(Twist())
            return

        now = time.time()
        if now - self.last_key_time > STOP_TIMEOUT:
            self.current_twist = Twist()

        self.pub.publish(self.current_twist)

    def destroy_node(self):
        self.running = False
        super().destroy_node()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--move-scale", type=float, default=1.0,
                        help="Scale factor for linear (x,y) motion speed")
    parser.add_argument("--rotate-scale", type=float, default=1.0,
                        help="Scale factor for rotation speed")
    args = parser.parse_args()

    # Kill pathFollower before starting keyboard control
    print("Starting keyboard control...")
    kill_path_follower()

    rclpy.init()
    node = KeyboardCmdVel(args.move_scale, args.rotate_scale)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.running = False
        node.destroy_node()
        
        # Restart pathFollower when exiting
        print("\nExiting keyboard control...")
        restart_path_follower()
        
        try:
            rclpy.shutdown()
        except:
            pass  # Ignore shutdown errors


if __name__ == "__main__":
    main()
