#!/bin/sh

cd `dirname $0`

n=3
m=3
r=1

id=1

u_mean=10
v_mean=10

samps=200
lookahead=200

procs=0

dir=../results/criteria/${n}x${m}_r${r}_u${u_mean}_v${v_mean}_${id}
echo $dir
mkdir -p $dir

datafile=$dir/data.pkl
apmffile=$dir/apmf.pkl
bayesfile=$dir/bayes_${samps}_${lookahead}.pkl


# first, generate data
if [[ -f $datafile ]]; then
    echo using existing data
else
    echo generating data
    python3 <<PYTHON
import pickle
import numpy as np
from active_pmf import make_fake_data

real, ratings, vals = make_fake_data(noise=0, mask_type='diag',
        num_users=$n, num_items=$m, rank=$r,
        u_mean=$u_mean, v_mean=$v_mean,
)

# get the entire first row except for one
ratings = np.vstack((ratings,
    [ [0, j, real[0, j]] for j in range(1, $n-1)]))

with open('$datafile', 'wb') as f:
    pickle.dump({'_real': real, '_ratings': ratings, '_vals': vals}, f)
PYTHON
fi


# get APMF criteria
echo
echo
echo
if [[ -f $apmffile ]]; then
    echo APMF results file already exists
else
    echo running variational
    ./active_pmf.py -s2 -D$r --load-data $datafile --save $apmffile -P $procs \
        pred-variance {pred-entropy-bound,uv-entropy,total-variance}{,-approx}
fi

# get MCMC criteria
echo
echo
echo
if [[ -f $bayesfile ]]; then
    echo bayesian results file already exists
else
    echo running bayesian
    ./run_bayes_pmf.py -s2 -D$r --load-data $datafile --save $bayesfile -P $procs \
        --samps $samps --lookahead-samps $lookahead --no-subtract-mean \
        pred-variance exp-variance
fi

# plot results
../plot_results.py --real --outdir $dir $datafile
../plot_results.py --criteria-firsts --kind bayes --outdir $dir $bayesfile
../plot_results.py --criteria-firsts --kind apmf --outdir $dir $apmffile
