FROM osrf/ros:humble-desktop as base
ENV ROS_DISTRO=humble
SHELL ["/bin/bash", "-c"]

# Set the working directory in the container
WORKDIR /project

# Copy the content of the local src directory to the working directory
COPY . /project

# Install dependencies
RUN apt-get update -y && \
    apt-get install -y doxygen && \
    apt-get clean

# Clone and build AriaCoda
RUN git clone https://github.com/reedhedges/AriaCoda.git
RUN cd AriaCoda && make && make install

# Build the base Colcon workspace, installing dependencies first.
RUN . /opt/ros/humble/setup.bash && \
    rosdep install --from-paths src --ignore-src --rosdistro humble -y && \
    colcon build --symlink-install

# Set the command to run the application
CMD ["gcc" , "src/ariaNode.cpp", "-o", "output.exe"]
RUN ["output.exe"]
