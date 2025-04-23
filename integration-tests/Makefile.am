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


INPUTS := h2o n2 hf
TMPFILE := $(shell mktemp)

all:
	@echo "Using tmpfile: $(TMPFILE)"
	@for input in $(INPUTS); do \
		$(MAKE) run CONFIG=$$input TMPFILE=$(TMPFILE); \
	done
	@echo ""
	@echo "==================== SUMMARY ===================="
	@cat $(TMPFILE)
	@rm -f $(TMPFILE)
	@echo ""
	@echo "Note: Some tests may fail if CTF is not available."
	@echo "      Without CTF all calculations use disk reader,"
	@echo "      but not all data is in the expected format."

run: clone
	@echo "+++ Running config: $(CONFIG).inp"
	@bash launcher.sh "$(TMPFILE)" "$(CONFIG).inp"

