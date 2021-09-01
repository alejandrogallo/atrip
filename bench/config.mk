BENCH_SOURCES = $(wildcard $(ATRIP_ROOT)/bench/test*.cxx)
BENCH_TARGETS = $(patsubst %.cxx,%,$(BENCH_SOURCES))

$(BENCH_TARGETS): CXXFLAGS += -I.
$(BENCH_TARGETS): CXXFLAGS += -fopenmp
bench-clean:
	-rm -v $(BENCH_TARGETS)
