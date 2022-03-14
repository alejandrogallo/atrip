include ./etc/make/emacs.mk

SUBDIRS = src bench

if WITH_BUILD_CTF
CTF_COMMIT         = 968f8f9eb6aab1d6b67d2fcc1a70c9fc3b98adfa
CTF_GIT_REPOSITORY = https://github.com/cc4s/ctf
CTF_BUILD_PATH     = $(builddir)/extern/build/$(CONFIG)/ctf/$(CTF_COMMIT)
CTF_SRC_PATH       = $(builddir)/extern/src/$(CONFIG)/ctf/$(CTF_COMMIT)
include $(srcdir)/etc/make/ctf.mk
src: ctf
SUBDIRS += extern/ctf/
endif

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
