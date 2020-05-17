FROM python:3.6-stretch as builder

ENV LANG C.UTF-8
# Env for python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Ensures that the python and pip executables used
# in the image will be those from our virtualenv.
ENV PATH="/venv/bin:$PATH"
# Define working directory
WORKDIR /home/app
# Setup env variables for training
ENV IS_CONLL_DATA=1
ENV USE_SCHEDULER=0
ENV USE_PRETRAINED_EMBEDDINGS=1
ENV NER_TRAIN_DATA=/home/app/data/conll2003/conll2003.train
ENV NER_TEST_DATA=/home/app/data/conll2003/conll2003.test
# Install OS package dependencies.
# Do all of this in one RUN to limit final image size.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gettext build-essential mariadb-client libmariadbclient-dev \
        libxml2-dev libxslt1-dev libxslt1.1 && \
    # Delete apt-get cache
    rm -rf /var/lib/apt/lists/*
# Copy file with dependencies first
COPY requirements.txt .
# Define user to run models
# and install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git htop vim && \
    # Delete apt-get cache
    rm -rf /var/lib/apt/lists/* && \
    adduser --system -u 1111 app && \
    chown app: -R /home/app && \
    python3 -m venv venv && \
    . ./venv/bin/activate && \
    pip install --no-cache-dir -r requirements.txt
# Then copy everything else
COPY . .
# Define user to run docker container
USER app
