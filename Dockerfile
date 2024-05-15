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

#install wget & pip
RUN apt install wget && wget https://bootstrap.pypa.io/get-pip.py && python3 ./get-pip.py 

#install depthai and opencv
RUN pip3 install depthai-sdk && pip3 install opencv-python 
# && pip3 install --upgrade setuptools && pip3 install ez_setup && pip3 install os-sys

#install joy and teleop
RUN apt install ros-humble-joy-linux ros-humble-teleop-twist-joy ros-humble-phidgets-spatial ros-humble-camera-info-manager ros-humble-rviz2 xvfb libxcb-xinerama0 ros-humble-slam-toolbox ros-humble-robot-localization -y

# RUN git clone -b master https://github.com/SICKAG/sick_scan_xd.git
# RUN cd test/scripts && chmod a+x ./*.bash && ./makeall_ros2_no_ldmrs.bash

RUN apt-get install --reinstall libqt5core5a libqt5gui5 libqt5widgets5
# Copy the content of the local src directory to the working directory
COPY . /project

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
