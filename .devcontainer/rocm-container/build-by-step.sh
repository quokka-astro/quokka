docker build --platform linux/amd64 -t quokka-linux-amd64-rocm:development .
docker tag quokka-linux-amd64-rocm:development ghcr.io/quokka-astro/quokka-linux-amd64-rocm:development 
docker push ghcr.io/quokka-astro/quokka-linux-amd64-rocm:development 
