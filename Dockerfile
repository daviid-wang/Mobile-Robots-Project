# Set base image (host OS)
FROM ubuntu:22.04

# Copy the dependencies file to the working directory
# COPY requirements.txt /project/requirements.txt

# Set the working directory in the container
WORKDIR /project

# Install any dependencies
# RUN pip install -r requirements.txt

# Copy the content of the local src directory to the working directory
COPY . /project

CMD ["python3" , "src/test.py"]