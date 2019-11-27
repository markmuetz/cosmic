import itertools
from pathlib import Path

import imageio
import numpy as np

def rmse(a1, a2):
    return np.sqrt(((a1 - a2)**2).mean())

if __name__ == '__main__':
    cwd = Path.cwd()
    comp_paths = list(cwd.glob('figs/*/'))

    for comp_path0, comp_path1 in itertools.combinations(comp_paths, 2):
        s0 = set(p.relative_to(comp_path0).as_posix() for p in comp_path0.glob('**/*.png'))
        s1 = set(p.relative_to(comp_path1).as_posix() for p in comp_path1.glob('**/*.png'))

        if s0 - s1:
            print(f'Only in {comp_path0}: {s0 - s1}')
        elif s1 - s0:
            print(f'Only in {comp_path1}: {s1 - s0}')
        else:
            print(f'All images in both {comp_path0} and {comp_path1}')

        common = s0 & s1

        for rel_path in common:
            full_path0 = comp_path0 / rel_path
            full_path1 = comp_path1 / rel_path

            im0 = imageio.imread(str(full_path0))
            im1 = imageio.imread(str(full_path1))
            try:
                if np.all(im0 == im1):
                    print(f'{comp_path0.parts[-1]} vs {comp_path1.parts[-1]}: Identical: {rel_path}')
                else:
                    print(f'{comp_path0.parts[-1]} vs {comp_path1.parts[-1]}: RMSE: {rmse(im0, im1)}: {rel_path}')
            except Exception as e:
                print(f'ERROR COMPARING {full_path0} and {full_path1}')
                print(e)


