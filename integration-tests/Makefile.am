REPO_URL = https://github.com/airmler/atrip_resources
REPO_DIR = atrip_resources
COMMIT = 8a6b37f

.PHONY: all clone run-h2o run-hf run-n2

# Global clone + checkout
clone:
	@if [ ! -d "$(REPO_DIR)/.git" ]; then \
		echo "Cloning repo..."; \
		git init $(REPO_DIR) \
		cd $(REPO_DIR) \
		git remote add origin $(REPO_URL) \
		git fetch --depth=1 origin $(COMMIT) \
		git checkout $(COMMIT) \
	else \
		echo "Repo already cloned."; \
	fi

run-h2o: clone
	bash run-h2o.sh

run-hf: clone
	bash run-hf.sh

run-n2: clone
	bash run-n2.sh

all: run-h2o run-hf run-n2


