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
    apt-get install -y doxygen build-essential 
    # && \
    # apt-get clean

# Clone and build AriaCoda
RUN git clone https://github.com/reedhedges/AriaCoda.git
RUN cd AriaCoda && make && make install

# install wget & pip
RUN apt install wget && wget https://bootstrap.pypa.io/get-pip.py && python3 ./get-pip.py 
#install depthai and opencv
# RUN pip install numpy
# RUN pip install easyocr
# RUN pip install matplotlib 
# RUN pip install collection
# RUN pip3 install opencv-python

# && pip3 install --upgrade setuptools && pip3 install ez_setup && pip3 install os-sys

#install joy and teleop
RUN apt install ros-humble-joy-linux ros-humble-teleop-twist-joy ros-humble-depthai-ros ros-humble-phidgets-drivers ros-humble-camera-info-manager ros-humble-sick-scan-xd ros-humble-rviz2 libxcb-xinerama0 ros-humble-slam-toolbox ros-humble-robot-localization ros-humble-joint-state-publisher ros-humble-joint-state-publisher-gui -y
#instal apt install ros-humble-std-msgs
RUN apt install ros-humble-std-msgs
RUN apt install ros-humble-navigation2 ros-humble-nav2-bringup -y
RUN apt install python3-rosbag -y
# RUN git clone https://github.com/gaia-platform/rosbag2_snapshot.gitx
# RUN cd src/
# RUN cd joy
RUN apt-get install --reinstall libqt5core5a libqt5gui5 libqt5widgets5
# Copy the content of the local src directory to the working directory
COPY . /project

RUN export ROS_DOMAIN_ID=153


RUN echo "xhost +" >> ~/.bashrc

# Source ROS environment and install dependencies for the project
# RUN . /opt/ros/humble/setup.bash && \
#     rosdep update && \
#     rosdep install --from-paths src --ignore-src --rosdistro=$ROS_DISTRO -y

RUN rosdep install --from-paths src --ignore-src --rosdistro=$ROS_DISTRO -y

# Build the ROS 2 workspace
RUN . /opt/ros/humble/setup.bash && \
    colcon build

RUN source /opt/ros/humble/setup.bash
RUN source /project/install/setup.bash
# RUN . install/setup.bash

RUN chmod +x /project/entrypoint.sh

ENTRYPOINT [ "/project/entrypoint.sh" ]
# RUN echo "source /project/install/setup.bash" >> ~/.bashrc
