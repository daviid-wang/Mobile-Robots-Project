# Mobile-Robots-Project
For working with the pioneer robots for AUTO4508

1. ssh group7@<ip address>
2. cd Desktop
3. git clone https://github.com/daviid-wang/Mobile-Robots-Project.git
4. docker build -t mobile-robots-project .
5. docker run -it mobile-robots-project
6. docker compose up (or) docker compose build 

To remove containers:
1. docker container prune
2. docker images
3. docker rmi <IMAGE_ID>

1. docker run -it mobile-robots-project
2. ros2 run ariaNode ariaNode -rp /dev/ttyUSB0