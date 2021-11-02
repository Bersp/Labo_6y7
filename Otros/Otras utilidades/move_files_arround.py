import itertools
import os
import re
import shutil

# -----------------------------------------------------------------------------
# Mover deformed del disco a la carpeta Mediciones

# n_fold_from = [
#     (re.findall(r'MED(\d+) - ', p)[0], p)
#     for p in sorted(os.listdir('.')) if 'MED' in p
# ]
# 
# n_fold_to = [
#     (re.findall(r'MED(\d+) - ', p)[0], p)
#     for p in sorted(os.listdir('../')) if 'MED' in p
# ]
# 
# from_to = [(f[1], f'../{t[1]}/deformed') for (f, t) in itertools.product(
#            n_fold_from, n_fold_to) if f[0] == t[0]]
# 
# for pair in from_to:
#     print(f'Pasando de {pair[0]} a {pair[1]}')
#     shutil.move(*pair) 

# -----------------------------------------------------------------------------
# Mover deformed del disco a la carpeta Mediciones
path = '../../Mediciones/MED44 - Bajada en voltaje - 1007/'
folders_to_copy = [f'{path}grey', f'{path}reference', f'{path}white']
destination_folders = [
    f'../../Mediciones/{p}/' for p in sorted(os.listdir('../../Mediciones/')) if 'MED' in p and
    45 <= int(re.findall(r'MED(\d+) - ', p)[0]) <= 62
]
for f, d in itertools.product(folders_to_copy, destination_folders):
    d = d + f.split('/')[-1]
    print(f'{f} -> {d}')
    shutil.copytree(f, d)
