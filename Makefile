include atrip.mk
CONFIG ?= gcc
PREFIX ?= /usr
SOURCES_FILE := Sources.mk

include $(SOURCES_FILE)
include ./etc/make/emacs.mk
include ./etc/config/$(CONFIG).mk
include ./bench/config.mk

$(info ==ATRIP== using configuration CONFIG=$(CONFIG))

ORG_MAIN = atrip.org
OBJ_FILES = $(patsubst %.cxx,%.o,$(filter-out %.hpp,$(ATRIP_SOURCES)))
DEP_FILES = $(patsubst %.o,%.d,$(OBJ_FILES))
ATRIP_SHARED_LIBRARY = lib/$(CONFIG)/libatrip.so
ATRIP_STATIC_LIBRARY = lib/$(CONFIG)/libatrip.a


extern: $(EXTERNAL_DEPENDENCIES)
clean-extern: CLEANING=yes
clean-extern:
	rm -vrf extern
#$(DEP_FILES): extern
.PHONY: extern

lib: extern
lib: $(DEP_FILES)
lib: $(ATRIP_SHARED_LIBRARY) $(ATRIP_STATIC_LIBRARY)
static: $(ATRIP_STATIC_LIBRARY)
shared: $(ATRIP_SHARED_LIBRARY)
.PHONY: lib static shared

ifeq ($(MAKECMD),lib)
include $(DEP_FILES)
endif



$(ATRIP_SHARED_LIBRARY): $(OBJ_FILES)
	mkdir -p $(@D)
	$(CXX) -shared $< $(CXXFLAGS) $(LDFLAGS) -o $@

$(ATRIP_STATIC_LIBRARY): $(OBJ_FILES)
	mkdir -p $(@D)
	$(AR) rcs $@ $<

$(SOURCES_FILE): config.el
	echo -n "ATRIP_SOURCES = " > $@
	$(EMACS) --eval '(atrip-print-sources)' >> $@

print:
	$(info $(filter-out %.hpp,$(ATRIP_SOURCES)))

$(ATRIP_SOURCES): $(ORG_MAIN)
	$(call tangle,$<)

tangle: $(ATRIP_SOURCES)

clean-emacs: CLEANING=yes
clean-emacs:
	-rm -v $(ATRIP_SOURCES)

clean: CLEANING=yes
clean:
	-rm -v $(OBJ_FILES) $(DEP_FILES)

clean-all: CLEANING=yes
clean-all: bench-clean clean-emacs clean clean-extern

bench: $(BENCH_TARGETS)

.PHONY: clean tangle bench

%: %.o
	$(info [bin] $@)
	$(CXX) $< $(CXXFLAGS) $(LDFLAGS) -o $@

%.o: %.cxx
	$(info [obj] $@)
	$(CXX) -c $< $(CXXFLAGS) -o $@

%.d: %.cxx
	$(info [dep] $@)
	$(CXX) -M $< $(CXXFLAGS) -o $@

.PHONY: install
install:
	mkdir -p $(PREFIX)/include
	cp -r include/* $(PREFIX)/include/
	mkdir -p $(PREFIX)/lib
	cp $(wildcard $(ATRIP_SHARED_LIBRARY) $(ATRIP_STATIC_LIBRARY)) $(PREFIX)/lib/

.PHONY: html
EMACS_HTML = $(EMACS) --load ./etc/emacs/html.el
html:
	$(EMACS_HTML) $(ORG_MAIN) --eval "(org-html-export-to-html)"
