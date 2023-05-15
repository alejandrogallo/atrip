include $(top_srcdir)/etc/make/emacs.mk

SUBDIRS = src bench test extern

all: atrip

_ATRIPSRCS = Sources.mk

ORG_MAIN = atrip.org

tangle: $(ORG_MAIN)
	$(call tangle,$<)

clean-emacs:
	-rm -v $(_ATRIPSRCS)

.PHONY: dbg tangle extern all atrip
dbg: src/dbg.h
include/dbg.h:
	wget -O $@ "https://raw.githubusercontent.com/sharkdp/dbg-macro/master/dbg.h"


atrip: extern
	$(MAKE) -C src

extern:
	$(MAKE) -C extern
