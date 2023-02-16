# Pull official base image
FROM python:3.10.8-bullseye

# Set working directory
RUN mkdir -p /usr/src/vectorisation
WORKDIR /usr/src

# Add source code
COPY . vectorisation

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update \
  && apt-get -y install netcat gcc \
  && apt-get clean

# Install python dependencies
RUN pip install --upgrade pip
RUN pip install -r vectorisation/requirements.txt
