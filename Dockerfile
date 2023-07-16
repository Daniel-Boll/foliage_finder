# Use an official Tensorflow runtime as a parent image
FROM tensorflow/tensorflow:latest-gpu-jupyter

# Set the working directory to /app
WORKDIR /app

# Add local directory's content to the docker image under /app 
ADD . /app

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

RUN apt-get update && apt-get install -y \
    wget \
    libcairo2-dev \
    libgl1-mesa-glx \
    python3-tk

# Install any needed packages specified in requirements.txt
COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

# RUN echo 'complete -W "`grep -oE '\''^[a-zA-Z0-9_.-]+:([^=]|$)'\'' Makefile | sed '\''s/[^a-zA-Z0-9_.-]*$//'\`" make' >> /root/.bashrc
