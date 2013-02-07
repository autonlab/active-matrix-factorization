.PHONY: tidy clean

STAN_HOME ?= $(HOME)/misc/stan-src-1.1.1
TOP := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

%.stan.model.pkl2.gz: %.stan
	python2 -c 'from rstan_interface import get_model; get_model("$<")'
%.stan.model.pkl3.gz: %.stan
	python3 -c 'from rstan_interface import get_model; get_model("$<")'

%: %.stan
	make -C $(STAN_HOME) $(TOP)/$@

tidy:
	rm -f *.d *.o bpmf*.cpp

clean: tidy
	rm -f bpmf *.stan.model.pkl?.gz
