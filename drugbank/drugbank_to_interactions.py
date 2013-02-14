#!/usr/bin/env python
import numpy as np
from lxml import objectify

def get_interactions(drugs):
    pid_to_idx = dict((pid, idx) for idx, pid in enumerate(
        p.attrib['id'] for p in drugs.partners.partner))

    interactions = np.zeros((len(drugs.drug), len(pid_to_idx)), int)

    for i, drug in enumerate(drugs.drug):
        if (len(drug.targets) > 0
                and getattr(drug.targets, 'target', None) is not None):
            for target in drug.targets.target:
                interactions[i, pid_to_idx[target.get('partner')]] = 1

    # kill any all-zero rows or columns
    return interactions[np.ix_(interactions.any(axis=1),
                               interactions.any(axis=0))]

if __name__ == '__main__':
    import bz2
    import gzip

    with bz2.BZ2File('drugbank.xml.bz2', 'rb') as f:
        drugbank = objectify.parse(f)
    interactions = get_interactions(drugbank.getroot())

    with gzip.GzipFile('drugbank_interactions.npy.gz', 'wb') as f:
        np.save(f, interactions)
