FROM python:3.6-stretch

ENV LANG C.UTF-8
# Env for python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Ensures that the python and pip executables used
# in the image will be those from our virtualenv.
ENV PATH="/venv/bin:$PATH"
# Define working directory
WORKDIR /home/app
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
    apt-get install git && \
    apt-get install htop && \
    apt-get install vim && \
    # Delete apt-get cache
    rm -rf /var/lib/apt/lists/* && \
    adduser --system -u 1111 app && \
    chown app: -R /home/app && \
    python3 -m venv venv && \
    . ./venv/bin/activate && \
    pip install --no-cache-dir -r requirements.txt
# Copy allentune for HyperParameters Search
RUN git clone https://github.com/allenai/allentune.git
# Then copy everything else
COPY . .
# Define user to run docker container
USER app

CMD ["./train.sh"]
