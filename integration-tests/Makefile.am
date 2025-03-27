H2O:
	git clone --depth 2 https://github.com/airmler/atrip_resources H2O
	cd $@; git reset --hard 929a6b0be4ed20e431d4c51f11b3f59ef0b1981d

.PHONY: run-h2o run-hf all
run-h2o: H2O
	bash ./run-h2o.sh

HF:
	@if [ ! -d "atrip_resources/.git" ]; then \
		git clone --depth 1 https://github.com/airmler/atrip_resources; \
	else \
		echo "Directory 'atrip_resources' already exists. Skipping clone."; \
	fi
	cd atrip_resources/hf

run-hf: HF
	bash ./run-hf.sh

all: run-h2o run-hf
