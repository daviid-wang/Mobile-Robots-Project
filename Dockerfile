FROM osrf/ros:humble-desktop as base
ENV ROS_DISTRO=humble
SHELL ["/bin/bash", "-c"]

RUN mkdir project

# Set the working directory in the container
WORKDIR /project

# Copy the content of the local src directory to the working directory
COPY . /project

# Install dependencies
RUN apt-get update -y && \
    apt-get install -y doxygen build-essential && \
    apt-get clean

# Clone and build AriaCoda
RUN cd src && git clone https://github.com/reedhedges/AriaCoda.git
RUN cd src/AriaCoda && make && make install

# Source ROS environment and install dependencies for the project
RUN . /opt/ros/humble/setup.bash && \
    rosdep update && \
    rosdep install --from-paths src --ignore-src --rosdistro=$ROS_DISTRO -y

# Build the ROS 2 workspace
RUN . /opt/ros/humble/setup.bash && \
    colcon build --symlink-install


RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
RUN echo "source install/setup.bash" >> ~/.bashrc