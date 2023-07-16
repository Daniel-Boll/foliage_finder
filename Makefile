.PHONY: build run clean clean_augmented help

help: ## Show this help
	@echo "Available targets:"
	@echo
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

build: ## Build the Docker image
	docker-compose build

run: ## Run the Docker container
	docker-compose run -u $(shell id -u):$(shell id -g) tensorflow bash

sudo_run: ## Run the Docker container as sudo
	docker-compose run tensorflow bash

drun: clean ## Run the code within the docker
	python src/main.py

clean: ## Clean output and classified directories
	rm -rf outputs/**
	rm -rf classified/**

clean_augmented: ## Clean augmented images
	./scripts/remove_augmented_images.bash
