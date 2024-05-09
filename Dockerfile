FROM osrf/ros:humble-desktop as base
ENV ROS_DISTRO=humble
SHELL ["/bin/bash", "-c"]

ARG USERNAME=group7
ARG USER_UID=152
ARG USER_GID=$USER_UID

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
RUN apt install wget && wget https://bootstrap.pypa.io/get-pip.py &&python3 ./get-pip.py 

#install depthai and opencv
RUN pip3 install depthai-sdk && pip3 install opencv-python

#install joy and teleop
RUN apt install ros-humble-joy-linux ros-humble-teleop-twist-joy -y
# RUN cd joy

# Copy the content of the local src directory to the working directory
COPY . /project

# Source ROS environment and install dependencies for the project
RUN . /opt/ros/humble/setup.bash && \
    rosdep update && \
    rosdep install --from-paths src --ignore-src --rosdistro=$ROS_DISTRO -y

# Build the ROS 2 workspace
RUN . /opt/ros/humble/setup.bash && \
    colcon build --symlink-install

RUN source /opt/ros/humble/setup.bash
RUN source /project/install/setup.bash
RUN . install/setup.bash

RUN chmod +x /project/entrypoint.sh

ENTRYPOINT [ "/project/entrypoint.sh" ]

# RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
# RUN echo "source /project/install/setup.bash" >> ~/.bashrc