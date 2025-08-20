ARG BASE_IMAGE=sarahwei0210/meshlrm:1.0.0

FROM $BASE_IMAGE

ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libstdc++6 \
    libgomp1 \
    libffi-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    tk-dev \
    uuid-dev \
    zlib1g-dev \
    xz-utils \
    ca-certificates \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /workspace

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install diso==0.1.4 \
        git+https://github.com/eliphatfs/cumesh2sdf.git \
        libigl==2.5.1 \
        numpy==1.26.4 \
        trimesh==4.4.0 \
        triton==2.0.0 \
        warp==1.0.4 \
        warp-lang==1.0.0b2 \
        git+https://github.com/seonghunn/pdmc.git

# Set environment variables
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6