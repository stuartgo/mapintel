#!/bin/bash

function join_by {
	local d=${1-} f=${2-}
	if shift 2; then
		printf %s "$f" "${@/#/$d}"
	fi
}

# Builds the images and tags them
dockerfiles=("Dockerfile-api-cpu" "Dockerfile-api-gpu" "Dockerfile-ui")
for dfile in "${dockerfiles[@]}"; do
	echo "####################################################################"
	echo "Building image from $dfile."
	echo "####################################################################"
	readarray -d "-" -t arr <<<"$dfile"
	arr[0]=mapintel
	docker build -f $dfile -t "ghcr.io/nova-ims-innovation-and-analytics-lab/$(join_by '-' ${arr[*]})" ..
done
