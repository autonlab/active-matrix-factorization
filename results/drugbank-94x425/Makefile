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

LATENT_D=20

.SECONDARY:

run%:
	mkdir -p $@

run%/*.log.gz: run%/*.log
	gzip $< > $@

run%/data.npz: | run%
	cd $(CODE_BASE) && $(PYTHON) choose_training.py \
		--drugbank \
		--n-pick 500 \
		--test-equal-classes --n-test 2000 \
		drugbank/subset_94x425.npy \
		$(RESULTS_BASE)/$@

run%/results_stan_nolookahead_s200w100_150steps.pkl: | run%/data.npz
	cd $(CODE_BASE)/stan-bpmf && $(GIT_SCRIPT) | head -n 1 | \
		xargs -I '{{REV}}' $(PY) bpmf.py \
		--note "git rev {{REV}}" \
		--latent-d $(LATENT_D) --discrete --subtract-mean \
		--model-filename bpmf_w0identity.stan \
		--samps 200 --warmup 100 \
		--steps 150 \
		--test-set-from-file --test-set crash-if-none-in-file \
		--procs $(PROCS) \
		--binary-acc \
		--load-data $(RESULTS_BASE)/$| --save-results $(RESULTS_BASE)/$@ \
		random pred-variance pred prob-ge-0 \
		$(OUTPUT)

# Don't subtract the mean, because of gradient descent problems
# but also because it's hardcoded :)
run%/results_mnpmf_nolookahead_150steps.pkl: | run%/data.npz
	cd $(CODE_BASE)/python-pmf && $(GIT_SCRIPT) | head -n 1 | \
		xargs -I '{{REV}}' $(PY) mn_active_pmf.py \
		--note "git rev {{REV}}" \
		--latent-d $(LATENT_D) --discrete-integration \
		--refit-lookahead \
		--processes $(PROCS) \
		--steps 150 \
		--load-data $(RESULTS_BASE)/$| --save-results $(RESULTS_BASE)/$@ \
		-- random pred-variance pred prob-ge-3.5 \
		$(OUTPUT)

run%/results_mmmf_150steps.pkl: | run%/data.npz
	cd $(CODE_BASE)/mmmf && $(GIT_SCRIPT) | head -n 1 | \
		xargs -I '{{REV}}' $(PY) active_mmmf.py \
		--note "git rev {{REV}}" \
		--data-file $(RESULTS_BASE)/$| --results-file $(RESULTS_BASE)/$@ \
		--matlab $(MATLAB) \
		--steps 150 \
		-C 1 \
		$(OUTPUT)
