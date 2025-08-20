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

# Clone PaMO repository
RUN git clone --recurse-submodules https://github.com/SarahWeiii/pamo.git /workspace/pamo

# Set work directory to PaMO
WORKDIR /workspace/pamo

# Install PaMO components
RUN echo "Installing stage 2: simplification..." \
    && cd simp_cuda \
    && pip install .

RUN echo "Installing stage 3: safe projection..." \
    && cd simp_cuda/safe_project/warp_ \
    && chmod +x ./tools/packman/packman \
    && python build_lib.py --cuda_path /usr/local/cuda \
    && pip install . \
    && cd .. \
    && pip install . \
    && cd ../..

# Set the default working directory to PaMO
WORKDIR /workspace/pamo

# Create a simple entry point script
RUN echo '#!/bin/bash\n\
echo "PaMO environment is ready!"\n\
echo "You can now run: python example.py --input INPUT_DIR --output OUTPUT_DIR --ratio 0.001"\n\
echo "Or use the demo: bash demo.sh"\n\
exec "$@"' > /entrypoint.sh \
    && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]