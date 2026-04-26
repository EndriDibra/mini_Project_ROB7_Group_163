# Socially Aware Robotic Path Planning using Computer Vision, A* Search, and Vision-Language Models

## Authors
**Endri Dibra and Daniel Dharampal**

## Project Overview
This project was developed as part of the **Advanced Robotic Perception** course (ROB7 – Group 163). It focuses on generating robot navigation paths that are not only efficient and collision-free, but also **socially aware**, safe, and human-friendly.

Traditional path planners usually optimize for shortest distance. In contrast, this system introduces a higher level of reasoning by combining:

- **Computer Vision** for obstacle detection  
- **A\*** path planning for optimal navigation  
- **Vision-Language Models (VLMs)** for social reasoning  
- **Path smoothing techniques** for realistic robot motion  

The final result is a robot trajectory that avoids obstacles, respects human-centered spaces, and moves in a predictable and polite manner.

---

## Core Technologies Used

### Computer Vision
The system uses **YOLOv11n** to detect dynamic and static obstacles such as:

- People  
- Chairs  
- Furniture  
- Other scene objects  

Detected objects are converted into an occupancy grid and inflated safety zones to preserve comfortable distance.

### Vision-Language Intelligence
The project integrates **SmolVLM-Instruct** as a social reasoning model.

The VLM analyzes the environment image and estimates:

- Socially inappropriate areas  
- Crowded or uncomfortable zones  
- Regions near seated people or obstacles  
- Better navigation corridors  

This creates a **Social Cost Map** that influences robot decisions beyond geometry alone.

### Path Planning
Navigation is solved using a customized **A\*** search algorithm enhanced with:

- Obstacle proximity penalties  
- Social cost penalties  
- Turn penalties  
- Curvature penalties  

This allows generation of smoother and more natural robot paths.

### Path Optimization
The raw path is refined through:

- **Line-of-Sight pruning** → removes unnecessary waypoints  
- **B-Spline interpolation** → smooth continuous trajectory generation  

---

## Full Pipeline

1. Load mapped environment image  
2. Detect people and obstacles using YOLO  
3. Build obstacle inflation map  
4. Query VLM for global social cost estimation  
5. Query VLM again for local object threat scoring  
6. Merge physical + social costs into unified map  
7. Run enhanced A* planner  
8. Prune redundant nodes  
9. Smooth final path with splines  
10. Re-evaluate generated path with VLM scoring system  

---

## Output

The system generates:

- Final path drawn on original map  
- Social cost heatmap  
- Navigation quality evaluation  
- Social awareness score  
- Speed / efficiency score  

---

## Main File

### `A_Star_Path_Planning.py`

This file contains the complete implementation including:

- Object detection  
- Cost map generation  
- Social reasoning logic  
- A* planning  
- Path smoothing  
- Visualization  

Other Python files in the repository are modular subsets created for easier understanding.

---

## Research Extension: SmolVLA

This repository also contains extra research work involving **SmolVLA**, focused on robotic learning from demonstration datasets.

### Included Scripts

#### `download_data_aria2.py`
Efficient multi-threaded downloader for SCAND rosbag datasets using `aria2c`.

#### `format_dataset.py`
Converts ROS bag files into **LeRobot-compatible datasets** with:

- Multi-camera temporal frames  
- Train/test split  
- Dynamic robot instructions  
- State-action formatting  

#### `finetune_notes.py`
Training configurations and shell commands for SmolVLA fine-tuning using progressive optimization strategies.

#### `test.py`
Evaluation pipeline for trained SmolVLA policies using:

- MSE  
- MAE  
- Trajectory consistency  
- Prediction horizon analysis  

---

## Why This Project Matters

Robots operating around humans must do more than avoid collisions.

They must also:

- Respect personal space  
- Avoid disturbing people  
- Move predictably  
- Behave naturally in shared environments  

This project explores how **Large Vision Models + Classical Robotics** can be combined to solve that challenge.

---

## Future Improvements

Possible next steps:

- Real-time ROS2 deployment  
- Dynamic moving pedestrian prediction  
- Reinforcement learning refinement  
- Multi-robot cooperation  
- 3D LiDAR + RGB fusion  
- Simulation in Gazebo / Isaac Sim  

---

## Final Note

This project reflects my interest in merging:

- Artificial Intelligence  
- Computer Vision  
- Robotics  
- Human-Robot Interaction  
- Autonomous Systems  

with practical engineering solutions.
