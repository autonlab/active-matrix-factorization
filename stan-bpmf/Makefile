.PHONY: tidy clean

STAN_HOME ?= $(HOME)/misc/stan-src-1.1.0
TOP := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

%: %.stan
	make -C $(STAN_HOME) $(TOP)/$@

tidy:
	rm -f *.d *.o bpmf.cpp

clean: tidy
	rm -f bpmf
