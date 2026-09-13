**SLAM/CV setup plan for RL-BOT**

Scope: simulation first, with interfaces that can later accept physical sensors. The original ROS 2 Jazzy / SLAM Toolbox architecture is retained. `ros2_ws/src/rlbot_bridge/` now provides the ROS bridge, mapping/localization launch, and map-saving services client; the Dockerfile runs the stack on Ubuntu 24.04. `src/rlbot/sensors.py` supplies raw and tilt-gated projected LiDAR, and `src/rlbot/odometry.py` supplies wheel/gyro dead reckoning. Local checks and a real Docker mapping/save/restart/localization test pass. Robot assets and cameras remain unchanged; turning required correcting yaw feedback and heading-dependent pitch measurements. Nav2 navigation and object CV remain unimplemented. Hardware models, onboard compute, and the demo deadline are still unspecified.

First demo: manually map the room, save and reload the map, navigate to a table, and report a colored cube's measured 3D position. Add a camera-guided grasp after perception and docking pass independently.

**What the repository already provides**

| Component | Observed state | Work needed |
| --- | --- | --- |
| Simulation | MuJoCo room with three tables, objects, walls, pillar, and divider | Repeatable sensor and navigation scenarios |
| Balance | Heading-independent pitch stabilization, bounded speed/yaw commands, acceleration limits and watchdog | Nav2 integration and hardware calibration |
| Cameras | `head_cam`, `wrist_left_cam`, and `wrist_right_cam` already defined by `scripts/build_mjcf.py` | Image/depth capture, calibration, transforms, and visibility checks |
| Lidar | Raw ray-cast scans and tilt-gated projected ROS scans | Validate physical mount and wider operating conditions |
| Motion sensors | Wheel/gyro odometry, IMU, joints, simulation clock and TF published to ROS | Calibrated covariance and accelerometer/attitude fusion |
| Mapping | SLAM Toolbox mapping, map/pose-graph save, and localization restart pass in Docker | Broader mapping routes and navigation integration |
| Manipulation | IK and scripted grasp evaluation | Consume perceived targets; improve grasp reliability separately |

The README's camera checklist is behind the code. `robot.py` reads simulator orientation and velocity directly, but the new odometry estimator does not: it takes only wheel angles and gyro samples. The existing balance checks were rerun for the local-input milestone (BracketBot, a 300 N shove, and the toy model); this does not validate navigation.

**Proposed stack**

Use MuJoCo for physics and sensing, ROS 2 Jazzy for message transport, SLAM Toolbox for a 2D map, Nav2 for navigation, and OpenCV plus depth for the first object detector. SLAM Toolbox supports mapping, saved pose graphs, and localization. [SLAM Toolbox documentation](https://github.com/SteveMacenski/slam_toolbox)

Use an Ubuntu 24.04 environment for ROS 2 Jazzy; its documented binary targets include x86-64 and ARM64. Simulation and ROS now run together in the Ubuntu 24.04 Docker image; this Mac uses the dedicated `colima-rlbot` Docker context. Local sensor/odometry checks remain runnable without ROS. Camera rendering needs a separate smoke test; LiDAR mapping does not require graphics. [ROS 2 Jazzy platform documentation](https://docs.ros.org/en/jazzy/Installation/Alternatives/Ubuntu-Install-Binary.html)

Dependencies to provision during implementation: the existing MuJoCo/NumPy requirements; OpenCV; ROS packages for `rclpy`, `tf2_ros`, sensor/navigation/clock messages, `cv_bridge`, `robot_localization`, `slam_toolbox`, Nav2, RViz2, and rosbag2; plus colcon/rosdep for the bridge package. Check the existing MuJoCo pin and Python/NumPy compatibility with ROS image bindings before locking the environment. A discrete GPU is optional for the initial CV baseline; validate an available rendering backend first.

1. **Make the environment and sensor capture reproducible.**

   Load `models/room_scene.xml`, run the existing balance evaluation, and capture RGB plus metric depth from all three cameras. Check head-camera coverage at each table; its current view may need a downward tilt. Confirm wrist views see the approach and fingertips. Implement changes in the model builder and regenerate its output.

   Extend the existing `src/rlbot/sensors.py` with camera capture and add `scripts/view_sensors.py`. Start with 640 x 480 images at 15 Hz; the local recorder already captures configurable 360-degree lidar at 10 Hz. Treat these as profiling targets. Build lidar rays from the sensor pose, select scene geometry deliberately, and mask robot geometry without accidentally removing furniture. MuJoCo provides ray-intersection APIs for this implementation. [MuJoCo API reference](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html)

   Pass when saved images show the targets, depth matches known test distances, and scans match walls at several headings. Include no-return, self-occlusion, and tilted-base cases. Keep rendering outside the balance loop, using a consistent state snapshot; the model's 0.002-second step implies 500 control updates per simulated second.

2. **Establish timestamps, coordinate frames, and odometry.**

   The ROS bridge package under `ros2_ws/src/rlbot_bridge/` uses `src/rlbot/odometry.py` and publishes `/clock`, `/scan`, `/scan_raw`, `/imu/data`, `/odom`, joint states and TF with simulation timestamps. Camera RGB/depth/CameraInfo and rosbag recording remain follow-up work; the local numeric recording/replay workflow is available now.

   Define `map -> odom -> base_footprint -> base_link -> sensors`. The odometry estimator owns `odom -> base_footprint`; the attitude/kinematics publisher owns the leaning chassis and moving wrist transforms; SLAM owns `map -> odom`. Each transform gets one publisher. Nav2 requires continuous odometry and a consistent transform chain. [Nav2 odometry guide](https://docs.nav2.org/rolling/configuration_and_development/first_time_robot_setup_guide/odom/setup_odom/)

   Integrate wheel encoders using measured wheel radius, spacing, and joint signs, then fuse appropriate IMU measurements. Account for chassis pitch rate when converting relative wheel rotation to ground motion. Keep yaw drift realistic and handle acceleration/gravity conventions explicitly. Publish simulator truth separately for evaluation; an initial truth-based bridge smoke test is allowed but does not count as SLAM validation.

   Pass when straight travel, in-place turns, and a closed route produce consistent estimated motion without missing transforms. Derive camera intrinsics from the actual projection and resolution; verify MuJoCo-to-ROS optical-axis conversion and depth back-projection against known 3D points.

3. **Build and reload a room map.**

   Configure SLAM Toolbox with odometry and scans; begin with a 5 cm occupancy grid. First validate using recorded data, then drive a slow teleoperated loop with repeated views of corners, pillar, and divider. Save both the occupancy map and serialized SLAM state, and test localization after a restart.

   Resolve the balancing-robot issue explicitly: the mounted lidar pitches with the chassis. Transform returns with timestamped attitude, reject floor/ceiling hits, and quantify the error before feeding a projected scan to 2D SLAM. Projection cannot recover geometry that a tilted beam never observed. If this fails, limit mapping to sufficiently upright motion or evaluate a stabilized mount/3D sensing approach. A perfectly level virtual lidar may be a diagnostic baseline only.

   Proposed gate: after aligning estimated and true trajectories once at the start, final loop error below 10 cm and 5 degrees over three runs, with recognizable furniture and no persistent doubled walls. These are initial acceptance targets, not measured results.

4. **Detect and locate one object from images.**

   Add `src/rlbot/perception.py` and `scripts/evaluate_perception.py`. Start with a colored cube: HSV segmentation, a cleaned object mask, robust depth samples, and camera-to-base/map transforms at image time. OpenCV documents HSV range thresholding for this baseline. [OpenCV tutorial](https://docs.opencv.org/4.x/da/d97/tutorial_threshold_inRange.html)

   Output a tracked ID, class, confidence, timestamp, frame, estimated position, dimensions, and validity flag. A visible-surface depth centroid is not automatically the object center or a grasp pose; estimate geometry using the mask, table plane, and supported shape assumptions. Use a wrist view for final approach refinement.

   Pass when the first cube is detected in at least 90% of a saved set of visible, reachable views and its estimated grasp target is within 2 cm of the defined reference target. Evaluate occlusion and missing depth separately. Randomize object poses to prevent dependence on `room.py`; simulator object poses and segmentation IDs are scoring/labeling data only. Extend to bowls, plates, and crates after this baseline; learned detection is a later decision based on failures.

5. **Connect navigation and table docking.**

   The yaw feedback sign and heading-dependent pitch measurement have been corrected. `DriveController` provides bounded speed/yaw commands, acceleration ramps and a timeout back to zero-speed balance. Local tests cover forward/reverse travel, turning, stopping and a driven loop; the ROS integration test drives a loop through `/cmd_vel`. Nav2 is not connected yet. Validate broader trajectories, stopping clearance and hardware gains before using it for autonomous navigation.

   Add navigation configuration and `scripts/navigate.py`. Begin with conservative speeds, stowed arms, and a footprint covering the robot's swept volume. Add depth-based obstacles because the low lidar can miss tabletops and overhangs. Measure stopping distance and use it to set clearance. Handle sensor loss and localization failure by stopping travel while maintaining balance.

   Use known table docking poses only for initial navigation integration. Then estimate the table edge and object approach from perception. Target coarse arrival within 10 cm/5 degrees, followed by local camera-guided docking within 2 cm/2 degrees or a stricter tolerance established by IK reach checks. Require collision-free arrival without falling across repeated routes.

6. **Demonstrate the complete perception-to-action handoff.**

   Sequence: load/localize -> navigate -> observe -> refine dock -> publish reachable grasp target. Log scans, images, estimated poses, detections, commands, and outcome. Provide one launch command and a replayable evaluation scenario.

   Then replace the grasp script's known object target with the perceived target and attempt one cube grasp. Track mapping, docking, perception, and grasp outcomes separately: the README currently reports no reliable scripted lifts, so a failed lift alone cannot diagnose a CV failure. VLA training and general room cleanup follow this milestone.

**Order and work breakdown**

Implement sensor capture first. Odometry/SLAM and stationary-camera CV can then proceed independently. Forward-speed control can also proceed once sensor timing is established. Navigation requires verified control plus localization; the integrated demonstration requires navigation plus CV. Keep these as six reviewable implementation changes matching the numbered steps. Commit logs/configuration and small evaluation fixtures, and ignore large recordings.

Mapping now has a runnable Docker acceptance check: `scripts/check_ros_mapping.py` launches the real ROS stack, drives a loop, saves the occupancy map and serialized graph, restarts in localization mode, and scores against separate simulator truth. Next work is broader routes and sensor-placement/calibration validation, then Nav2 integration. Camera RGB/depth capture remains a separate sensor/CV task. The local numeric recorder alone is not evidence of working SLAM.

For physical deployment, first inventory actual lidar/camera models, encoder access, IMU rate, onboard OS/compute, motor interface, and sensor mounts. Then verify ROS drivers, power/data bandwidth, intrinsics/extrinsics, wheel dimensions, measured mass and motor limits, and timing. Replace the simulation bridge with drivers while retaining downstream interfaces. Hardware purchases and transfer estimates depend on that inventory.
