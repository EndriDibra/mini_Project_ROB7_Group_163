# Mini Project regarding Advanced Robotic Perception course from ROB7-Group-163
## Brief Description
This project implements a socially aware path planner designed to navigate an autonomous agent—like robot from a start point to a goal point within a mapped environment, optimizing not just for distance, but also for social acceptability. The process starts by loading a map and using a YOLO (You Only Look Once) model [YOLOv11n] to detect dynamic obstacles like people and chairs. These detected objects are used to create an inflated cost map, ensuring the robot maintains a safe distance from physical boundaries.

Crucially, the system introduces a layer of intelligence by querying a Visual Language Model (VLM), specifically SmolVLM-Instruct, with the map image. The VLM acts as a "social consciousness," analyzing the scene and returning a social cost map, which assigns higher costs (penalties) to paths passing through areas deemed socially inappropriate (e.g., cutting too close to seating areas or through dense crowds). These two costs, physical obstacle avoidance and social cost, are combined into a single cost landscape. The core pathfinding is executed by a A* search algorithm, which is augmented with penalties for sharp turns and high curvature to ensure the generated path is smooth and robot-friendly.

Once the raw path is found, it is refined using a line-of-sight pruning algorithm to remove redundant waypoints and then smoothed using B-spline interpolation to create a fluid, continuous trajectory. Finally, the VLM is queried again to provide an evaluation of the generated path's social awareness and optimality, offering a quantifiable feedback loop. In short, the program aims to generate paths that are not only efficient and safe but also polite and predictable, making robot movement seamless and less disruptive in human-centric spaces. The end result is a visual representation of the path overlaid on the map.

### A_Star_Path_Planning.py
This is the program that contains the entire logic and functions for this project. The other python files are subset of the A_Star_Path_Planning.py, provided for easier comprehension of the code and implementation of it.

## Extra Research: SmolVLA

### download_data_aria2.py
Downloads SCAND dataset rosbag files from Texas Digital Library using aria2c for optimized parallel downloads. Features include resumable downloads, multiple connections per file (16 by default), and concurrent file downloads (8 by default). Automatically verifies downloaded files and handles interruptions gracefully. Requires aria2c to be installed (`sudo apt-get install aria2`).

### format_dataset.py
Converts ROS bag files into LeRobot-compatible datasets with temporal multi-camera support. Processes odometry and image data at 4 FPS, creating episodes of 30 frames each. Supports dynamic instruction generation based on robot trajectory or static instructions. Features configurable temporal camera offsets for historical frame integration, automatic train/test splitting, and comprehensive validation. Outputs state vectors containing current and past velocities, with 32-dimensional action vectors (2D velocity commands + padding).

### finetune_notes.py
Shell command collection for training and evaluating SmolVLA models on the SCAND dataset. Contains three progressive training configurations: initial training (20k steps, batch size 32), full dataset training (30k steps, batch size 16), and fine-tuning with custom optimizer settings (50k steps, AdamW with gradient clipping). Includes evaluation and dataset visualization commands. All outputs are logged to training_log.txt.

### test.py
Inference evaluation script for trained SmolVLA policies. Loads checkpointed models and runs predictions on test datasets, computing MSE/MAE metrics across action dimensions. Features per-dimension error analysis, prediction consistency checks, and temporal trajectory evaluation. Uses training statistics for proper normalization/denormalization of states and actions. Outputs comprehensive metrics including first-step errors, horizon variance, and sample predictions, saving results to NumPy archives for further analysis.
