============
Installation
============

To clone the Tasmania repository on your machine and place yourself on the branch :option:`BRANCH`, from within a terminal run:

.. code-block:: bash

   git clone https://github.com/eth-cscs/tasmania.git
   cd tasmania
   git checkout BRANCH

Then, enter the folder :option:`docker/` and type

.. code-block:: bash

   git clone https://github.com/eth-cscs/gridtools4py.git
   cd gridtools4py
   git checkout merge_ubbiali

to clone the GridTools4Py repository. **Note**: both Tasmania and GridTools4Py repositories are *private*, so you should be given access to them in order to accomplish the steps above.

We suggest to run any application relying on Tasmania inside a Docker_ container, spawn from a provided image. To create a local instance of the image, named :option:`tasmania`, from the repository root directory issue

.. code-block:: bash

   docker build --build-arg uid=$(id -u) -t tasmania .

Later, launch

.. code-block:: bash

   docker run --rm -dit --privileged -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $PWD:/tasmania tasmania

to run the container in detached mode. The flag :option:`-v $PWD:/tasmania` attaches the current directory to the container *volume*, so that the folder :option:`/tasmania` within the container will be in sync with the repository root directory. Instead, the flags :option:`--privileged -e DISPLAY -e XAUTHORITY:$XAUTHORITY -v /tmp/.X11-unix:/tmp/.X11-unix` are required to execute any graphical application within the container, and should then be used to, e.g., generate a plot via Matplotlib_ or run a Jupyter_ notebook. **Note**: we are conscious of the fact that granting a container privileged rights is consider bad practice. Yet, it is the easiest way (to our knowledge) to allow containerized applications access the host desktop environment.

When successful, the previous command returns a long sequence of letters and digits, representing the container identifier. This is required to gain shell access to the running container:

.. code-block:: bash

   docker exec -it CONTAINER_ID bash

Here, :option:`CONTAINER_ID` is the container identifier.

.. _Docker: https://www.docker.com/
.. _Jupyter: http://jupyter.org/
.. _Matplotlib: https://matplotlib.org/
