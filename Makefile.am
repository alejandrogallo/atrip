include $(top_srcdir)/etc/make/emacs.mk
include $(top_srcdir)/atrip.mk

SUBDIRS = src bench test


_ATRIPSRCS = Sources.mk

ORG_MAIN = atrip.org

tangle: $(ORG_MAIN)
	$(call tangle,$<)

clean-emacs:
	-rm -v $(_ATRIPSRCS)

.PHONY: dbg tangle
dbg: include/dbg.h
include/dbg.h:
	wget -O $@ "https://raw.githubusercontent.com/sharkdp/dbg-macro/master/dbg.h"
