import numpy as np
from lxml import objectify

def get_interactions(drugs):
    pid_to_idx = {pid: idx for idx, pid in enumerate(
        p.attrib['id'] for p in drugs.partners.partner)}

    interactions = np.zeros((len(drugs.drug), len(pid_to_idx)), int)

    for i, drug in enumerate(drugs.drug):
        if len(drug.targets) > 0:
            for target in drug.targets.target:
                interactions[i, pid_to_idx[target.get('partner')]] = 1

    return interactions


if __name__ == '__main__':
    with open('drugbank.xml') as f:
        drugbank = objectify.parse(f)
    interactions = get_interactions(drugbank.getroot())
