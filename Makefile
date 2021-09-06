DOXYGEN_INDEX = doxygen/html/index.html

all: $(DOXYGEN_INDEX)

$(DOXYGEN_INDEX):
	doxygen Doxyfile

