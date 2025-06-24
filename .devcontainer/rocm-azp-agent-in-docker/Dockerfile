# Use the AMD ROCm image as the base image. I build this on macOS (arm64).
FROM ghcr.io/quokka-astro/quokka-amd64-rocm-frontier:latest
ENV TARGETARCH="linux/amd64"

USER root
RUN zypper --non-interactive refresh && \
  zypper --non-interactive update && \
  zypper --non-interactive install curl git jq libicu74 python3-h5py && \
  zypper clean --all

# Install Azure CLI
RUN rpm --import https://packages.microsoft.com/keys/microsoft.asc && \
    zypper --non-interactive addrepo --name 'Azure CLI' --check https://packages.microsoft.com/yumrepos/azure-cli azure-cli && \
    zypper --non-interactive install azure-cli
		
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
