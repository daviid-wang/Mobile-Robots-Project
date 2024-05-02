# Set base image (host OS)
FROM osrf/ros:humble-desktop as base
ENV ROS_DISTRO=humble
SHELL ["/bin/bash", "-c"]

# Set the working directory in the container
WORKDIR /project

#copy ARIA and ARIA code

# Copy the content of the local src directory to the working directory
COPY . /project

RUN git clone https://github.com/cinvesrob/Aria.git

# Build the base Colcon workspace, installing dependencies first.
RUN . /opt/ros/humble/setup.bash && \
    apt-get update -y && \
    rosdep install --from-paths src --ignore-src --rosdistro humble -y && \
    colcon build --symlink-install

# CMD ["python3" , "src/test.py"]