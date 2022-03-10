# variables
#
SPHINX_BUILD_DIR = build
DOXYGEN_INDEX = $(SPHINX_BUILD_DIR)/doxygen/html/index.html

# rules
#
all: doxygen sphinx

.PHONY: sphinx doxygen

doxygen: $(DOXYGEN_INDEX)
$(DOXYGEN_INDEX):
	mkdir -p $(@D)
	doxygen Doxyfile

$(SPHINX_BUILD_DIR)/index.html: index.rst atrip.rst conf.py
	sphinx-build . $(SPHINX_BUILD_DIR)
sphinx: $(SPHINX_BUILD_DIR)/index.html

index.rst: ../README.rst
	mv $< $@

atrip.rst: ../atrip.rst
	mv $< $@

../%.rst:
	make -C $(dir $@) $(notdir $@)

clean:
	rm -rfv sphinx doxygen/ index.rst atrip.rst
