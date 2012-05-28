#!/usr/bin/env python3

from collections import defaultdict
import bz2
import numpy as np
import pickle

def counter(n=0):
    def inner():
        nonlocal n
        n += 1
        return n - 1
    return inner

server_ids = defaultdict(counter())
client_ids = defaultdict(counter())
bandwidths = defaultdict(list)

with bz2.BZ2File('PlanetLabTrace.txt.bz2') as f:
    next(f) # skip header
    for line in f:
        client, server, data_size, _, elapsed_time = line.decode().split(',')
        sid = server_ids[server]
        cid = client_ids[client]
        bandwidths[sid, cid].append(int(data_size) / int(elapsed_time) * 1000)

with open('id_mappings.pkl', 'wb') as f:
    pickle.dump({'server_ids': dict(**server_ids), 'client_ids': dict(**client_ids)}, f)

num_servers = server_ids['__blargh']
num_clients = client_ids['__blargh']
matrix = np.empty((num_servers, num_clients)); matrix.fill(np.nan)
for (i, j), b in bandwidths.items():
    matrix[i, j] = np.mean(b)
with bz2.BZ2File('bandwidths.npy.bz2', 'wb') as f:
    np.save(f, matrix)

known = np.isfinite(matrix)
good_rows = known.sum(axis=1) >= 10
good_cols = known.sum(axis=0) >= 10
with bz2.BZ2File('bandwidths_10plus.npy.bz2', 'wb') as f:
    np.save(f, matrix[good_rows,:][:,good_cols])
