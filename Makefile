# HELP
# This will output the help for each task
# thanks to https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
.PHONY: help

help: ## This help.
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.DEFAULT_GOAL := help


# DOCKER TASKS
# Build the container
build-cpu: ## Build the container
	docker build -t mtg-gpt2-cpu -f dev/docker/cpu/Dockerfile .

build-gpu: ## Build the container
	docker build -t mtg-gpt2-gpu -f dev/docker/gpu/Dockerfile .

run-cpu: ## Run the container
	docker run --rm -v $(shell pwd):$(shell pwd) -it mtg-gpt2-cpu:latest

run-gpu: ## Run the container
	nvidia-docker run --rm -v $(shell pwd):$(shell pwd) -it -p 8989:8989 mtg-gpt2-gpu:latest

run-jupyter: ## Run the container
	docker run --rm -v $(shell pwd):$(shell pwd) -it -p 8999:8999 mtg-gpt2-cpu:latest /bin/sh -c 'cd $(shell pwd); jupyter notebook --allow-root --no-browser --port=8999 --ip=0.0.0.0;'