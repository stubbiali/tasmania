#!/bin/bash

IMAGE_NAME=tasmania:master
CONTAINER_NAME=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 12 | head -n 1)

echo "About to fire up a containter named '$CONTAINER_NAME' from the image '$IMAGE_NAME'." 
read -n 1 -s -r -p "Press any key to continue, or Ctrl-C to exit."
echo ""

docker run --rm															\
		   --privileged													\
		   -dit															\
		   -e DISPLAY 													\
		   -e XAUTHORITY=$XAUTHORITY 									\
		   -e PYTHONPATH=/home/dockeruser/tasmania						\
		   -P															\
		   --name $CONTAINER_NAME										\
		   --mount type=bind,src=$PWD/..,dst=/home/dockeruser/tasmania	\
		   --mount type=bind,src=/tmp/.X11-unix,dst=/tmp/.X11-unix		\
		   $IMAGE_NAME													\
		   bash

docker exec -it				\
			$CONTAINER_NAME \
			bash -c "set -ex; \
					 curl -LO https://bootstrap.pypa.io/get-pip.py; \
					 python get-pip.py --user; \
					 cd tasmania; \
					 make distclean; \
					 python -m pip install --user -e .; \
					 cd ..; \
					 bash"
