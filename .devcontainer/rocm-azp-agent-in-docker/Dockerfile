# Use the AMD ROCm image as the base image
FROM ghcr.io/quokka-astro/quokka-amd64-rocm-frontier:latest
ENV TARGETARCH="linux/amd64"

USER root

# Install required packages including Python 3.11
RUN zypper --non-interactive refresh && \
    zypper --non-interactive update && \
    zypper --non-interactive install \
        curl \
        git \
        jq \
        libicu-devel \
        wget \
        tar \
        gzip \
        which \
        python311 \
        python311-pip \
        python311-devel \
        python311-setuptools \
        python311-wheel && \
    zypper clean --all

# Set Python 3.11 as the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/pip3 pip3 /usr/bin/pip3.11 1

# Install Python packages with the new Python version
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install h5py cached-property numpy matplotlib 

# Install Azure CLI for openSUSE
RUN rpm --import https://packages.microsoft.com/keys/microsoft.asc && \
    zypper --non-interactive addrepo \
        --name 'Azure CLI' \
        --check https://packages.microsoft.com/yumrepos/azure-cli \
        azure-cli && \
    zypper --non-interactive install azure-cli && \
    zypper clean --all

# Create directories and set permissions
WORKDIR /azp/

RUN groupadd agent && \
    useradd -m -d /home/agent -s /bin/bash -g agent agent && \
    usermod -a -G users agent && \
    chown -R agent:agent /azp /home/agent

# Switch to agent user
WORKDIR /home/agent
USER agent
# Another option is to run the agent as root.
# ENV AGENT_ALLOW_RUNASROOT="true"

# WORKDIR /home/agent
# RUN touch /home/agent/.token
# RUN chmod 666 /home/agent/.token

# # Do not use start.sh in the image. Put start.sh in the job folder and run it via 'singularity run ... ./start.sh'
# ENTRYPOINT [ "./start.sh" ]
