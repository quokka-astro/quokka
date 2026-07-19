export DOCKER_BUILDKIT=1
docker buildx create --use

docker buildx build --platform linux/amd64 \
  -t ghcr.io/quokka-astro/quokka-linux-amd64-rocm:development \
  -f ./Dockerfile \
  --push .
