# Pull official base image
FROM ubuntu/postgres:edge

# Create databases
COPY create.sql /docker-entrypoint-initdb.d

# Set working directory
RUN mkdir -p /usr/src/db
WORKDIR /usr/src

# Add source code
COPY . db

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update \
  && apt-get -y install python3 python3-pip netcat gcc \
  && apt-get clean

# Install python dependencies
RUN pip install --upgrade pip
RUN pip install -r db/requirements.txt

# Entrypoint
RUN chmod +x /usr/src/db/entrypoint.sh
ENTRYPOINT ["/usr/src/db/entrypoint.sh"]
