FROM osrf/ros:humble-desktop as base
ENV ROS_DISTRO=humble
SHELL ["/bin/bash", "-c"]

ARG USERNAME=group7
ARG USER_UID=152
ARG USER_GID=$USER_UID
RUN echo "xhost +" >> ~/.bashrc

RUN mkdir project

# Set the working directory in the container
WORKDIR /project

# Install dependencies
RUN apt-get update -y && \
    apt-get install -y doxygen build-essential ros-${ROS_DISTRO}-navigation2 ros-${ROS_DISTRO}-slam-toolbox ros-${ROS_DISTRO}-nav2-gazebo-spawner && \
    apt-get install -y python3-colcon-common-extensions && \
    apt-get clean

# Clone and build AriaCoda
RUN git clone https://github.com/reedhedges/AriaCoda.git
RUN cd AriaCoda && make && make install

# Install wget & pip
RUN apt install wget && wget https://bootstrap.pypa.io/get-pip.py && python3 ./get-pip.py 

# Install depthai and opencv
RUN pip install numpy && pip3 install opencv-python && pip install easyocr && pip install matplotlib && pip install collection

# Install other ROS packages
RUN apt install ros-humble-joy-linux ros-humble-teleop-twist-joy ros-humble-depthai-ros ros-humble-phidgets-drivers ros-humble-camera-info-manager ros-humble-sick-scan-xd ros-humble-rviz2 libxcb-xinerama0 ros-humble-joint-state-publisher ros-humble-joint-state-publisher-gui -y
RUN apt-get install --reinstall libqt5core5a libqt5gui5 libqt5widgets5

# Copy the content of the local src directory to the working directory
COPY . /project

RUN echo "xhost +" >> ~/.bashrc

# Source ROS environment and install dependencies for the project
RUN . /opt/ros/humble/setup.bash && \
    rosdep update && \
    rosdep install --from-paths src --ignore-src --rosdistro=$ROS_DISTRO -y

# Build the ROS 2 workspace
RUN . /opt/ros/humble/setup.bash && \
    colcon build --symlink-install

RUN source /opt/ros/humble/setup.bash
RUN source /project/install/setup.bash

RUN chmod +x /project/entrypoint.sh

ENTRYPOINT [ "/project/entrypoint.sh" ]
