from pathlib import Path
import pickle
from timeit import default_timer as timer

import iris
import pandas as pd


class FileStore:
    EXT_MAP = {
        '.pkl': 'pickle',
        '.nc': 'iris_cube',
        '.hdf': 'pandas_df',
    }

    def __init__(self, store_in_mem=True, create_dirs=True):
        self.store_in_mem = store_in_mem
        self.create_dirs = create_dirs
        self._obj_cache = {}
        self._record_log = []

    def __call__(self, filename, gen_fn=None,
                 objtype=None, load=None, write=None,
                 load_args=[], load_kwargs={},
                 save_args=[], save_kwargs={},
                 gen_fn_args=[], gen_fn_kwargs={}):

        filename = Path(filename)
        if not objtype:
            objtype = FileStore.EXT_MAP[filename.suffix]

        if objtype == 'custom':
            assert load and write, 'must provide load/write fns for custom objtype'

        start = timer()
        if filename in self._obj_cache:
            result = self._obj_cache[filename]
            load_type = 'mem'
        else:
            if filename.exists():
                if objtype == 'pickle':
                    result = pickle.load(filename.open('rb'), *load_args, **load_kwargs)
                elif objtype == 'iris_cube':
                    result = iris.load(str(filename), *load_args, **load_kwargs)
                elif objtype == 'pandas_df':
                    result = pd.read_hdf(str(filename), key=str(filename.stem), *load_args, **load_kwargs)
                elif objtype == 'custom':
                    result = load(str(filename), *load_args, **load_kwargs)
                load_type = 'file'
            else:
                if not gen_fn:
                    raise Exception(f'No value found in cache for {filename}')
                result = gen_fn(*gen_fn_args, **gen_fn_kwargs)
                if self.create_dirs and not filename.parent.exists():
                    filename.parent.mkdir(parents=True, exist_ok=True)

                if objtype == 'pickle':
                    pickle.dump(result, filename.open('wb'), *save_args, **save_kwargs)
                elif objtype == 'iris_cube':
                    iris.save(result, str(filename), *save_args, **save_kwargs)
                elif objtype == 'pandas_df':
                    result.to_hdf(str(filename), key=str(filename.stem), *save_args, **save_kwargs)
                elif objtype == 'custom':
                    write(result, str(filename), *save_args, **save_kwargs)
                load_type = 'gen'

            if self.store_in_mem:
                self._obj_cache[filename] = result

        end = timer()
        self._record_log.append((objtype, load_type, repr(result), end - start))
        return result

    def clear(self):
        self._obj_cache.clear()
