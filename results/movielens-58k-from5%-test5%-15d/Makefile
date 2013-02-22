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

LATENT_D=15

.SECONDARY:

run%:
	mkdir -p $@

run%/data.npz: | run%
	cd $(CODE_BASE) && $(PYTHON) choose_training.py \
		--pick-known-frac 0.05 \
		--test-at-random --test-known-frac 0.05 \
		movielens-100k/half_ratings_70.npy.gz \
		$(RESULTS_BASE)/$@

run%/results_stan_nolookahead_s200w100_200steps.pkl: | run%/data.npz
	cd $(CODE_BASE)/stan-bpmf && $(GIT_SCRIPT) | head -n 1 | \
		xargs -I '{{REV}}' $(PY) bpmf.py \
		--note "git rev {{REV}}" \
		--latent-d $(LATENT_D) --discrete --subtract-mean \
		--model-filename bpmf_w0identity.stan \
		--samps 200 --warmup 100 \
		--steps 200 \
		--test-set-from-file --test-set crash-if-none-in-file \
		--procs $(PROCS) \
		--load-data $(RESULTS_BASE)/$| --save-results $(RESULTS_BASE)/$@ \
		random pred-variance pred prob-ge-3.5 \
		$(OUTPUT)

# Don't subtract the mean, because of gradient descent problems
# but also because it's hardcoded :)
run%/results_mnpmf_nolookahead_200steps.pkl: | run%/data.npz
	cd $(CODE_BASE)/python-pmf && $(GIT_SCRIPT) | head -n 1 | \
		xargs -I '{{REV}}' $(PY) mn_active_pmf.py \
		--note "git rev {{REV}}" \
		--latent-d $(LATENT_D) --discrete-integration \
		--refit-lookahead \
		--processes $(PROCS) \
		--steps 200 \
		--load-data $(RESULTS_BASE)/$| --save-results $(RESULTS_BASE)/$@ \
		-- random pred-variance pred prob-ge-3.5 \
		$(OUTPUT)

run%/results_mmmf_200steps.pkl: | run%/data.npz
	cd $(CODE_BASE)/mmmf && $(GIT_SCRIPT) | head -n 1 | \
		xargs -I '{{REV}}' $(PY) active_mmmf.py \
		--note "git rev {{REV}}" \
		--data-file $(RESULTS_BASE)/$| --results-file $(RESULTS_BASE)/$@ \
		--matlab $(MATLAB) \
		-C 1 --cutoff 3.5 \
		--steps 200 \
		$(OUTPUT)

run%/results_mmmf_random_200steps.pkl: | run%/data.npz
	cd $(CODE_BASE)/mmmf && $(GIT_SCRIPT) | head -n 1 | \
		xargs -I '{{REV}}' $(PY) active_mmmf.py \
		--note "git rev {{REV}}" \
		--data-file $(RESULTS_BASE)/$| --results-file $(RESULTS_BASE)/$@ \
		--matlab $(MATLAB) \
		-C 1 --cutoff 3.5 \
		--steps 200 \
		random \
		$(OUTPUT)
run%/results_mmmf_min-margin_200steps.pkl: | run%/data.npz
	cd $(CODE_BASE)/mmmf && $(GIT_SCRIPT) | head -n 1 | \
		xargs -I '{{REV}}' $(PY) active_mmmf.py \
		--note "git rev {{REV}}" \
		--data-file $(RESULTS_BASE)/$| --results-file $(RESULTS_BASE)/$@ \
		--matlab $(MATLAB) \
		-C 1 --cutoff 3.5 \
		--steps 200 \
		min-margin \
		$(OUTPUT)
run%/results_mmmf_min-margin-pos_200steps.pkl: | run%/data.npz
	cd $(CODE_BASE)/mmmf && $(GIT_SCRIPT) | head -n 1 | \
		xargs -I '{{REV}}' $(PY) active_mmmf.py \
		--note "git rev {{REV}}" \
		--data-file $(RESULTS_BASE)/$| --results-file $(RESULTS_BASE)/$@ \
		--matlab $(MATLAB) \
		-C 1 --cutoff 3.5 \
		--steps 200 \
		min-margin-pos \
		$(OUTPUT)
run%/results_mmmf_max-margin_200steps.pkl: | run%/data.npz
	cd $(CODE_BASE)/mmmf && $(GIT_SCRIPT) | head -n 1 | \
		xargs -I '{{REV}}' $(PY) active_mmmf.py \
		--note "git rev {{REV}}" \
		--data-file $(RESULTS_BASE)/$| --results-file $(RESULTS_BASE)/$@ \
		--matlab $(MATLAB) \
		-C 1 --cutoff 3.5 \
		--steps 200 \
		max-margin \
		$(OUTPUT)
run%/results_mmmf_max-margin-pos_200steps.pkl: | run%/data.npz
	cd $(CODE_BASE)/mmmf && $(GIT_SCRIPT) | head -n 1 | \
		xargs -I '{{REV}}' $(PY) active_mmmf.py \
		--note "git rev {{REV}}" \
		--data-file $(RESULTS_BASE)/$| --results-file $(RESULTS_BASE)/$@ \
		--matlab $(MATLAB) \
		-C 1 --cutoff 3.5 \
		--steps 200 \
		max-margin-pos \
		$(OUTPUT)
