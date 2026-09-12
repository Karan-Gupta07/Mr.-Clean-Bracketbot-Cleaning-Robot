FROM ros:jazzy-ros-base-noble@sha256:386d06ec6d4188f731bae5678e07b4cb64a4e4d4152090c0bd1f881dcf7706f5

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-venv python3-colcon-common-extensions python3-rosdep \
    ros-jazzy-slam-toolbox ros-jazzy-nav2-map-server ros-jazzy-tf2-ros \
    libgl1 && apt-get clean

ENV RLBOT_REPO=/opt/rlbot
ENV PATH="/opt/venv/bin:${PATH}"
ENV PYTHONUNBUFFERED=1
ENV ROS_DOMAIN_ID=42
WORKDIR /opt/rlbot
COPY requirements.txt ./
RUN python3 -m venv --system-site-packages /opt/venv && \
    python -m pip install --no-cache-dir -r requirements.txt numpy==2.4.2 setuptools==79.0.1 && \
    python -m pip check
COPY . ./
RUN source /opt/ros/jazzy/setup.bash && \
    python /usr/bin/colcon --log-base ros2_ws/log build --base-paths ros2_ws/src \
    --build-base ros2_ws/build --install-base ros2_ws/install --symlink-install && \
    chmod +x docker/entrypoint.sh
ENTRYPOINT ["/opt/rlbot/docker/entrypoint.sh"]
CMD ["ros2", "launch", "rlbot_bridge", "mapping.launch.py"]
