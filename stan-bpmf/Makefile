.PHONY: tidy clean

STAN_HOME ?= $(HOME)/misc/stan-src-1.0.3
TOP := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

bpmf: bpmf.stan
	make -C $(STAN_HOME) $(TOP)/bpmf

tidy:
	rm -f *.d *.o bpmf.cpp

clean: tidy
	rm -f bpmf
