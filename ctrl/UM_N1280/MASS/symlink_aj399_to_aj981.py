"""
u-aj981 is the continuation of u-aj399.
I have downloaded these separately from MASS. This script symlinks
all files in u-aj399 to u-aj981.
"""
from pathlib import Path
from common import BASE_OUTPUT_DIRPATH

aj981_path = BASE_OUTPUT_DIRPATH / 'u-aj981/ap9.pp'
for symlink_target in aj981_path.glob('precip_??????/aj981a.p9????????.pp'):
    symlink = Path(str(symlink_target).replace('aj981', 'aj399'))
    if not symlink.exists():
        print(f'{symlink} -> {symlink_target}')
        symlink.parent.mkdir(exist_ok=True)
        symlink.symlink_to(symlink_target)
    else:
        assert symlink.is_symlink()
        print(f'{symlink} exists')

