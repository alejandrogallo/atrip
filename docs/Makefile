# variables
#
SPHINX_BUILD_DIR = build
DOXYGEN_INDEX = $(SPHINX_BUILD_DIR)/doxygen/html/index.html
EXTRA_STYLESHEET_URL = https://github.com/jothepro/doxygen-awesome-css.git
DX_HTML_EXTRA_STYLESHEET  = doxygen-awesome-css/doxygen-awesome.css

# rules
#
all: doxygen sphinx

.PHONY: sphinx doxygen

doxygen: $(DOXYGEN_INDEX)
$(DX_HTML_EXTRA_STYLESHEET):
	git clone $(EXTRA_STYLESHEET_URL)

$(DOXYGEN_INDEX): Doxyfile $(DX_HTML_EXTRA_STYLESHEET)
	mkdir -p $(@D)
	doxygen $<

$(SPHINX_BUILD_DIR)/index.html: index.rst atrip.rst conf.py
	sphinx-build . $(SPHINX_BUILD_DIR)
sphinx: $(SPHINX_BUILD_DIR)/index.html

index.rst: ../README.rst
	cp $< $@

atrip.rst: ../atrip.rst
	cp $< $@

../%.rst:
	make -C $(dir $@) $(notdir $@)

clean:
	rm -rfv sphinx doxygen/ index.rst atrip.rst
