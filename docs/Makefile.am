include $(top_srcdir)/etc/make/emacs.mk

# variables
#
SPHINX_BUILD_DIR = build
DOXYGEN_INDEX = $(SPHINX_BUILD_DIR)/doxygen/html/index.html

DX_EXTRA_STYLESHEET_URL = https://github.com/jothepro/doxygen-awesome-css.git
DX_HTML_EXTRA_FOLDER = $(builddir)/doxygen-awesome-css
DX_HTML_EXTRA_STYLESHEET = $(DX_HTML_EXTRA_FOLDER)/doxygen-awesome.css


all: doxygen sphinx
.PHONY: sphinx doxygen

doxygen: $(DOXYGEN_INDEX)

$(DOXYGEN_INDEX): Doxyfile $(DX_HTML_EXTRA_STYLESHEET)
	mkdir -p $(@D)
	doxygen $<

$(DX_HTML_EXTRA_STYLESHEET):
	git clone $(DX_EXTRA_STYLESHEET_URL) $(DX_HTML_EXTRA_FOLDER)

$(SPHINX_BUILD_DIR)/index.html: index.rst atrip.rst conf.py
	sphinx-build -b html . $(SPHINX_BUILD_DIR)
sphinx: $(SPHINX_BUILD_DIR)/index.html

index.rst: $(top_srcdir)/README.rst
	cp $< $@

atrip.rst: $(top_srcdir)/atrip.rst
	cp $< $@

clean-local:
	rm -rfv sphinx doxygen/ index.rst atrip.rst \
		$(SPHINX_BUILD_DIR) \
		$(DX_HTML_EXTRA_FOLDER)
