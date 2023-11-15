include $(top_srcdir)/etc/make/emacs.mk

SUBDIRS = src bench test extern integration-tests

all: atrip

_ATRIPSRCS = Sources.mk

ORG_MAIN = atrip.org

tangle: $(ORG_MAIN)
	$(call tangle,$<)

clean-emacs:
	-rm -v $(_ATRIPSRCS)

.PHONY: tangle extern all atrip

atrip: extern
	$(MAKE) -C src

extern:
	$(MAKE) -C extern
