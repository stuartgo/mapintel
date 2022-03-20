#!/bin/bash

# Parse flag u(username) and p(pass) arguments
while getopts u:p: flag; do
	case "${flag}" in
	u) username=${OPTARG} ;;
	p) pass=${OPTARG} ;;
	esac
done

# Execute build_images script
/bin/bash build_images.sh

# Authenticate to the container registry
echo $pass | docker login ghcr.io -u $username --password-stdin

# Pushing container images to registry
images=(
	"ghcr.io/nova-ims-innovation-and-analytics-lab/mapintel-ui:latest"
	"ghcr.io/nova-ims-innovation-and-analytics-lab/mapintel-api-cpu:latest"
	"ghcr.io/nova-ims-innovation-and-analytics-lab/mapintel-api-gpu:latest"
)

for i in "${images[@]}"; do
	echo "####################################################################"
	echo "Pushing image $i to registry."
	echo "####################################################################"
	docker push $i
done
