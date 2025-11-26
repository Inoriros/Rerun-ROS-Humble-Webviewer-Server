#!/bin/bash

# Define the name for the tmux session
SESSION_NAME="rerun_habitat"

# --- Setup: Clean Start ---
tmux kill-session -t $SESSION_NAME 2>/dev/null

# Create a new tmux session with the first window (Rerun server)
tmux new-session -d -s $SESSION_NAME -n rerun_server

# --- Window 0: Rerun Server ---
RERUN_CMD="pkill -f 'rerun .*--web-viewer' || true && rerun --serve-web --web-viewer --bind 0.0.0.0 --port 9876 --web-viewer-port 9090"
tmux send-keys -t $SESSION_NAME:0 "$RERUN_CMD" C-m


# --- Window 1: Habitat → Rerun Python Script ---
tmux new-window -t $SESSION_NAME -n habitat_bridge
# PYTHON_CMD="export ROS_DOMAIN_ID=88 && python3 habitatsim_to_rerun_nobridge.py"
PYTHON_CMD="python3 habitatsim_to_rerun_nobridge.py"
tmux send-keys -t $SESSION_NAME:1 "$PYTHON_CMD" C-m


# --- Window 2: Keyboard cmd_vel Controller ---
tmux new-window -t $SESSION_NAME -n teleop
# TELEOP_CMD="export ROS_DOMAIN_ID=88 && python3 keyboard_cmd_vel.py --move-scale 1.0 --rotate-scale 1.0"
TELEOP_CMD="python3 keyboard_cmd_vel.py --move-scale 1.0 --rotate-scale 1.0"
tmux send-keys -t $SESSION_NAME:2 "$TELEOP_CMD" C-m


# --- Final Step: Attach to Session ---
echo "Starting processes in tmux session '$SESSION_NAME'."
echo "Use Ctrl+b followed by window number (0,1,2) to switch."

tmux attach-session -t $SESSION_NAME


# Instruction to view the Rerun web viewer:
# If opening directly on the server via tunnel:
#    http://localhost:9090/
# Choose: Open from URL → rerun+http://127.0.0.1:9876/proxy
