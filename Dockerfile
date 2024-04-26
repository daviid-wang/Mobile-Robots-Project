# Set base image (host OS)
FROM osrf/ros:humble-desktop as base
ENV ROS_DISTRO=humble
SHELL ["/bin/bash", "-c"]

# Copy the dependencies file to the working directory
# COPY requirements.txt /project/requirements.txt

# Set the working directory in the container
WORKDIR /project

# Copy the content of the local src directory to the working directory
COPY . /project

RUN . /opt/ros/humble/setup.bash && \
    apt-get update -y && \
    rosdep install --from-paths src --ignore-src --rosdistro humble -y && \
    colcon build --symlink-install

# Install any dependencies
# RUN pip install -r requirements.txt

# CMD ["python3" , "src/test.py"]