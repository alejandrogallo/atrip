H2O:
	git clone --depth 1 https://github.com/airmler/atrip_resources H2O
	cd $@; git reset --hard 929a6b0be4ed20e431d4c51f11b3f59ef0b1981d

.PHONY: run-h2o all
run-h2o: H2O
	bash ./run-h2o.sh


all: run-h2o
