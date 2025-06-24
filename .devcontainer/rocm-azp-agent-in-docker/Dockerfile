# Use the AMD ROCm image as the base image
FROM ghcr.io/quokka-astro/quokka-amd64-rocm-frontier:latest
ENV TARGETARCH="linux/amd64"

USER root

# Install required packages
RUN zypper --non-interactive refresh && \
    zypper --non-interactive update && \
    zypper --non-interactive install \
        curl \
        git \
        jq \
        libicu-devel \
        python3-h5py \
        wget \
        tar \
        gzip \
        which && \
    zypper clean --all

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

# Create agent user with proper groups
RUN useradd -m -d /home/agent -s /bin/bash agent && \
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
