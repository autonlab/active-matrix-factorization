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

LATENT_D=1

.SECONDARY:

data.pkl:
	$(PY) gen.py $@

results_stan.pkl: data.pkl
	cd $(CODE_BASE)/stan-bpmf && $(GIT_SCRIPT) | head -n 1 | \
		xargs -I '{{REV}}' $(PY) bpmf.py \
		--note "git rev {{REV}}" \
		--latent-d $(LATENT_D) --no-discrete --no-subtract-mean \
		--test-set all \
		--samps 200 --warmup 100 \
		--lookahead-samps 100 --lookahead-warmup 50 \
		--initialize-at-pmf-map \
		--hyperparams '{"mu_0": np.array([10])}' \
		--num-integration-pts 25 \
		--steps 2 \
		--procs $(PROCS) \
		--load-data $(RESULTS_BASE)/$< --save-results $(RESULTS_BASE)/$@ \
		pred-variance exp-variance exp-entropy-est \
		$(OUTPUT)

results_mnpmf.pkl: data.pkl
	cd $(CODE_BASE)/python-pmf && $(GIT_SCRIPT) | head -n 1 | \
		xargs -I '{{REV}}' $(PY) mn_active_pmf.py \
		--note "git rev {{REV}}" \
		--latent-d $(LATENT_D) --continuous \
		--refit-lookahead \
		--steps 2 \
		--processes $(PROCS) \
		--load-data $(RESULTS_BASE)/$< --save-results $(RESULTS_BASE)/$@ \
		pred-variance total-variance total-variance-approx \
		uv-entropy uv-entropy-approx \
		$(OUTPUT)

results_apmf.pkl: data.pkl
	cd $(CODE_BASE)/python-pmf && $(GIT_SCRIPT) | head -n 1 | \
		xargs -I '{{REV}}' $(PY) active_pmf.py \
		--note "git rev {{REV}}" \
		--latent-d $(LATENT_D) --continuous \
		--refit-lookahead \
		--steps 2 \
		--processes $(PROCS) \
		--load-data $(RESULTS_BASE)/$< --save-results $(RESULTS_BASE)/$@ \
		pred-variance total-variance total-variance-approx \
		uv-entropy uv-entropy-approx \
		pred-entropy-bound pred-entropy-bound-approx \
		$(OUTPUT)

results_rc.pkl: data.pkl
	cd $(CODE_BASE)/ratingconcentration && $(GIT_SCRIPT) | head -n 1 | \
		xargs -I '{{REV}}' $(PY) active_rc.py \
		--notes "git rev {{REV}}" \
		--data-file $(RESULTS_BASE)/$< --results-file $(RESULTS_BASE)/$@ \
		--matlab $(MATLAB) \
		--delta 1.5 --pred-mean \
		--steps 2 \
		entropy random \
		$(OUTPUT)

%/firsts.png: results_%.pkl
	mkdir -p $*
	cd $(CODE_BASE) && $(PY) ./plot_results.py \
		--outdir $(RESULTS_BASE)/$* \
		--criteria-firsts \
		--cmap YlOrRd \
		$(RESULTS_BASE)/$<
