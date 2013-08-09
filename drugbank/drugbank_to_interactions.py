#!/usr/bin/env python
import numpy as np
from lxml import objectify

def get_interactions(drugs):
    pid_to_idx = dict((pid, idx) for idx, pid in enumerate(
        p.attrib['id'] for p in drugs.partners.partner))

    interactions = np.zeros((len(drugs.drug), len(pid_to_idx)), dtype=bool)
    drug_names = np.array([str(d.name) for d in drugs.drug])
    target_ids = np.array([int(p.attrib['id']) for p in drugs.partners.partner])

    for i, drug in enumerate(drugs.drug):
        if (len(drug.targets) > 0
                and getattr(drug.targets, 'target', None) is not None):
            for target in drug.targets.target:
                interactions[i, pid_to_idx[target.get('partner')]] = True


    # kill any all-zero rows or columns
    good_drug = interactions.any(axis=1)
    good_partner = interactions.any(axis=0)
    
    good_inters = interactions[np.ix_(good_drug, good_partner)]
    return good_inters, drug_names[good_drug], target_ids[good_partner]

if __name__ == '__main__':
    import argparse
    import bz2
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', default='drugbank.xml.bz2')
    parser.add_argument('outfile', nargs='?', default='drugbank_interactions.npz')
    args = parser.parse_args()

    with bz2.BZ2File(args.infile, 'rb') as f:
        drugbank = objectify.parse(f)
    interactions, drug_names, target_ids = get_interactions(drugbank.getroot())

    np.savez_compressed(args.outfile,
        interactions=interactions,
        drug_names=drug_names,
        target_ids=target_ids)
