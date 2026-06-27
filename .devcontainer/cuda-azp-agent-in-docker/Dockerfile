# Use the CUDA image as the base image. I build this on macOS (arm64).
FROM ghcr.io/quokka-astro/quokka-linux-amd64-cuda:development
ENV TARGETARCH="linux/amd64"

USER root
RUN apt update && \
  apt upgrade -y && \
  apt install -y curl git jq libicu74 python3-h5py gfortran ccache

# Install h5py
# RUN pip3 install h5py 

# Install Azure CLI
RUN curl -sL https://aka.ms/InstallAzureCLIDeb | bash

WORKDIR /azp/

# COPY ./start.sh ./
# RUN chmod +x ./start.sh

# Create agent user and set up home directory
RUN useradd -m -d /home/agent agent
RUN chown -R agent:agent /azp /home/agent

WORKDIR /home/agent
USER agent
# Another option is to run the agent as root.
# ENV AGENT_ALLOW_RUNASROOT="true"

# WORKDIR /home/agent
# RUN touch /home/agent/.token
# RUN chmod 666 /home/agent/.token

# # Do not use start.sh in the image. Put start.sh in the job folder and run it via 'singularity run ... ./start.sh'
# ENTRYPOINT [ "./start.sh" ]
