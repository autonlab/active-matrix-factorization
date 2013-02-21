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
	PDB_ERR = --pdb-on-error
else
	PY = $(PYTHON)
	IGNORE_GIT ?= abort
	PDB_ERR = --no-pdb-on-error
endif
GIT_SCRIPT = $(CODE_BASE)/get_git_rev.sh $(IGNORE_GIT)


# TODO: smarter formulae/formats that split up results for each key, but
#		know how to run multiple keys in parallel

# NOTE: this makefile mostly uses | dependencies to avoid reruns based
#       on modification time of the data file, to avoid very expensive ops
#       just in case. if you want to rerun, just delete the data files.

LATENT_D=10

.SECONDARY:

run%:
	mkdir -p $@

run%/data.npz: | run%
	cd $(CODE_BASE) && $(PYTHON) choose_training.py \
		--new-item-frac 0.1 \
		--pick-no-extras \
		--test-at-random --test-known-frac 0.05 \
		movielens-100k/half_ratings_70.npy.gz \
		$(RESULTS_BASE)/$@

run%/data.npz.initfit_s200w200.npz: | run%/data.npz
	cd $(CODE_BASE)/stan-bpmf && $(GIT_SCRIPT) | head -n 1 | \
		xargs -I '{{REV}}' $(PY) bpmf_newitems.py \
		--note "git rev {{REV}}" \
		$(PDB_ERR) \
		--latent-d $(LATENT_D) --subtract-mean \
		--initial-fit-only \
		--initial-fit-file $(RESULTS_BASE)/$@ \
		--initial-fit-samps 200 \
		--initial-fit-warmup 200 \
		--load-data $(RESULTS_BASE)/$|
		$(OUTPUT)

run%/results_stan_newitems_nolookahead_s200w100_200steps.pkl: run%/data.npz.initfit_s200w200.npz | run%/data.npz
	cd $(CODE_BASE)/stan-bpmf && $(GIT_SCRIPT) | head -n 1 | \
		xargs -I '{{REV}}' $(PY) bpmf_newitems.py \
		--note "git rev {{REV}}" \
		$(PDB_ERR) \
		--latent-d $(LATENT_D) --discrete --subtract-mean \
		--model-filename bpmf_newitems_w0identity.stan \
		--initial-fit-file $(RESULTS_BASE)/$< \
		--initial-fit-samps 200 \
		--initial-fit-warmup 200 \
		--samps 200 --warmup 100 \
		--steps 200 \
		--test-set-from-file --test-set crash-if-none-in-file \
		--procs $(PROCS) \
		--load-data $(RESULTS_BASE)/$| --save-results $(RESULTS_BASE)/$@ \
		-- random pred-variance pred prob-ge-3.5 \
		$(OUTPUT)


run%/results_stan_nolookahead_s200w100_200steps.pkl: | run%/data.npz
	cd $(CODE_BASE)/stan-bpmf && $(GIT_SCRIPT) | head -n 1 | \
		xargs -I '{{REV}}' $(PY) bpmf.py \
		--note "git rev {{REV}}" \
		$(PDB_ERR) \
		--latent-d $(LATENT_D) --discrete --subtract-mean \
		--model-filename bpmf_w0identity.stan \
		--samps 200 --warmup 100 \
		--steps 200 \
		--test-set-from-file --test-set crash-if-none-in-file \
		--query-new-only \
		--procs $(PROCS) \
		--load-data $(RESULTS_BASE)/$| --save-results $(RESULTS_BASE)/$@ \
		-- random pred-variance pred prob-ge-3.5 \
		$(OUTPUT)


run%/results_mmmf_200steps.pkl: | run%/data.npz
	cd $(CODE_BASE)/mmmf && $(GIT_SCRIPT) | head -n 1 | \
		xargs -I '{{REV}}' $(PY) active_mmmf.py \
		--note "git rev {{REV}}" \
		--data-file $(RESULTS_BASE)/$| --results-file $(RESULTS_BASE)/$@ \
		--matlab $(MATLAB) \
		--steps 200 \
		-C 1 --cutoff 3.5 \
		$(OUTPUT)
