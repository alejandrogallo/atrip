EMACS = emacs -q --batch
define tangle
$(EMACS) $(1) --eval "(require 'org)" --eval '(org-babel-tangle)'
endef

MAIN = README.org
SOURCES = $(shell grep -oe ':tangle \+[^ ]\+' $(MAIN) | awk '{print $$2}' | sort -u)

$(SOURCES): $(MAIN)
	$(call tangle,$<)

tangle: $(SOURCES)

clean:
	rm -r $(SOURCES)

.PHONY: clean tangle
