ATRIP_ROOT := $(PWD)
CONFIG ?= gcc
include etc/config/$(CONFIG).mk
include ./bench/config.mk

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
	-rm -v $(SOURCES)

clean-all: bench-clean clean

bench: $(BENCH_TARGETS)

.PHONY: clean tangle bench

%: %.o
	$(CXX) $< $(CXXFLAGS) $(LDFLAGS) -o $@

%.o: %.cxx
	$(CXX) -c $< $(CXXFLAGS) -o $@

