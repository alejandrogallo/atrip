DOXYGEN_INDEX = doxygen/html/index.html

all: $(DOXYGEN_INDEX) sphinx

$(DOXYGEN_INDEX):
	doxygen Doxyfile

.PHONY: sphinx
SPHINX_BUILD_DIR = sphinx
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
