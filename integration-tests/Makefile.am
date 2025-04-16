REPO_URL = https://github.com/airmler/atrip_resources
REPO_DIR = atrip_resources
COMMIT = 8a6b37f

.PHONY: all clone run-h2o run-hf run-n2

# Global clone + checkout
clone:
	@if [ ! -d "$(REPO_DIR)/.git" ]; then \
		echo "Cloning repo..."; \
		git clone $(REPO_URL) $(REPO_DIR); \
		cd $(REPO_DIR); \
		git checkout $(COMMIT); \
	else \
		echo "Repo already cloned."; \
	fi



TMPFILE := $(shell mktemp)

all:
	@echo "Using tmpfile: $(TMPFILE)"
	@$(MAKE) run-h2o TMPFILE=$(TMPFILE)
	@$(MAKE) run-hf TMPFILE=$(TMPFILE)
	@$(MAKE) run-n2 TMPFILE=$(TMPFILE)
	@echo ""
	@echo "====== SUMMARY ======"
	@cat $(TMPFILE)
	@rm -f $(TMPFILE)
	@echo "Note: Some tests may fail if CTF is not available."
	@echo "      Without CTF all calculations use disk reader,"
	@echo "      but not all data is in the expected format."

run-h2o: clone
	bash run-h2o.sh $(TMPFILE)

run-hf: clone
	bash run-hf.sh $(TMPFILE)

run-n2: clone
	bash run-n2.sh $(TMPFILE)
#all: run-h2o run-hf run-n2


