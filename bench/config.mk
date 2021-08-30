BENCH_SOURCES = $(wildcard $(ATRIP_ROOT)/bench/test*.cxx)
BENCH_TARGETS = $(patsubst %.cxx,%,$(BENCH_SOURCES))

bench-clean:
	-rm -v $(BENCH_TARGETS)
