# Use Python3.6 runtime as parent image
FROM python:3.6.4-jessie

# Set the working directory to /tasmania
WORKDIR /tasmania

# Copy the tasmania's root directory into the container at /tasmania
ADD . /tasmania

# Install required Python packages specified in requirements.txt
RUN pip install -r requirements.txt

# Add bootstrap file into the container
ADD bootstrap.sh .

# Run the bootstrap file
RUN chmod +x bootstrap.sh
RUN ./bootstrap.sh
