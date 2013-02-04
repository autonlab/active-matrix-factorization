cython=cython

.PHONY: all buld_exts clean cfiles

all: build_exts

build_exts: cfiles
	python3 setup.py build_ext --inplace

cfiles: normal_exps_cy.c pmf_cy.c bayes_pmf.c

normal_exps_cy.c: normal_exps_cy.pyx
	${cython} normal_exps_cy.pyx

pmf_cy.c: pmf_cy.pyx pmf_cy.pxd
	${cython} pmf_cy.pyx

bayes_pmf.c: bayes_pmf.py bayes_pmf.pxd
	${cython} -Xbinding=false bayes_pmf.py
# https://groups.google.com/forum/?fromgroups=#!topic/cython-users/QHStfaxywC8

clean:
	rm -f {normal_exps_cy,pmf_cy,bayes_pmf}.{c,so}
