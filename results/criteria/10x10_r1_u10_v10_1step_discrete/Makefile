# absolute paths to results and code dirs
CODE_BASE ?= $(HOME)/matrix-fact
RESULTS_BASE ?= $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

INTERACTIVE ?= 1

# use all cores by default
PROCS ?= 0

PYTHON ?= python3
MATLAB ?= matlab

OUTPUT = 2>&1 | tee $(RESULTS_BASE)/$@.log
ifeq "$(INTERACTIVE)" "1"
	PY = $(PYTHON) -u
	IGNORE_GIT ?= interactive
else
	PY = $(PYTHON)
	IGNORE_GIT ?= abort
endif
GIT_SCRIPT = $(CODE_BASE)/get_git_rev.sh $(IGNORE_GIT)


# TODO: smarter formulae/formats that split up results for each key, but
#		know how to run multiple keys in parallel

# NOTE: this makefile uses | dependencies everywhere to avoid reruns based
#       on modification time of the data file, to avoid very expensive ops
#       just in case. if you want to rerun, just delete the data files.

LATENT_D=2

.SECONDARY:

data.pkl:
	$(PY) gen.py $@

stan.pkl: data.pkl
	cd $(CODE_BASE)/stan-bpmf && $(GIT_SCRIPT) | head -n 1 | \
		xargs -I '{{REV}}' $(PY) bpmf.py \
		--note "git rev {{REV}}" \
		--latent-d $(LATENT_D) --discrete --no-subtract-mean \
		--test-set all \
		--samps 200 --warmup 100 \
		--lookahead-samps 100 --lookahead-warmup 50 \
		--steps 2 \
		--procs $(PROCS) \
		--load-data $(RESULTS_BASE)/$< --save-results $(RESULTS_BASE)/$@ \
		$(OUTPUT)

mnpmf.pkl: data.pkl
	cd $(CODE_BASE)/python-pmf && $(GIT_SCRIPT) | head -n 1 | \
		xargs -I '{{REV}}' $(PY) mn_active_pmf.py \
		--note "git rev {{REV}}" \
		--latent-d $(LATENT_D) --discrete \
		--refit-lookahead \
		--steps 2 \
		--processes $(PROCS) \
		--load-data $(RESULTS_BASE)/$< --save-results $(RESULTS_BASE)/$@ \
		$(OUTPUT)

apmf.pkl: data.pkl
	cd $(CODE_BASE)/python-pmf && $(GIT_SCRIPT) | head -n 1 | \
		xargs -I '{{REV}}' $(PY) active_pmf.py \
		--note "git rev {{REV}}" \
		--latent-d $(LATENT_D) --discrete \
		--refit-lookahead \
		--steps 2 \
		--processes $(PROCS) \
		--load-data $(RESULTS_BASE)/$< --save-results $(RESULTS_BASE)/$@ \
		$(OUTPUT)

mmmf.pkl: data.pkl
	cd $(CODE_BASE)/mmmf && $(GIT_SCRIPT) | head -n 1 | \
		xargs -I '{{REV}}' $(PY) active_mmmf.py \
		--note "git rev {{REV}}" \
		--data-file $(RESULTS_BASE)/$< --results-file $(RESULTS_BASE)/$@ \
		--matlab $(MATLAB) \
		-C 1 --cutoff 3.5 \
		--steps 2 \
		$(OUTPUT)

rc.pkl: data.pkl
	cd $(CODE_BASE)/ratingconcentration && $(GIT_SCRIPT) | head -n 1 | \
		xargs -I '{{REV}}' $(PY) active_rc.py \
		--notes "git rev {{REV}}" \
		--data-file $(RESULTS_BASE)/$< --results-file $(RESULTS_BASE)/$@ \
		--matlab $(MATLAB) \
		--delta 1.5 --pred-mean \
		--steps 2 \
		entropy random \
		$(OUTPUT)

%.png: %.pkl
	cd $(CODE_BASE) && $(PY) ./plot_results.py 
