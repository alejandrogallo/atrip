include ./etc/make/emacs.mk

SUBDIRS = src bench

_ATRIPSRCS = Sources.mk
## include $(_ATRIPSRCS)

ORG_MAIN = atrip.org
ATRIP_SHARED_LIBRARY = lib/$(CONFIG)/libatrip.so
ATRIP_STATIC_LIBRARY = lib/$(CONFIG)/libatrip.a

tangle: $(ORG_MAIN)
	$(call tangle,$<)

clean-emacs: CLEANING=yes
clean-emacs:
	-rm -v $(_ATRIPSRCS)

.PHONY: tangle

EMACS_HTML = $(EMACS) --load ./etc/emacs/html.el

%.html: %.org
	$(EMACS_HTML) $< -f org-html-export-to-html

%.rst: %.org
	$(EMACS_HTML) $< -f org-rst-export-to-rst

.PHONY: dbg
dbg: include/dbg.h
include/dbg.h:
	wget -O $@ "https://raw.githubusercontent.com/sharkdp/dbg-macro/master/dbg.h"
