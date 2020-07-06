# UM N1280 ctrl

Note, the files need to be downloaded from MASS first. See MASS subdir for this.

Control files for UM N1280 sims. The files in this directory are used to process the N1280 files to a) convert them from pp to nc and b) extract a region from them (possibly with combining rain/snow to precip and dealing with different variables, e.g. u-ak543 has stratiform rain (Large-Scale) rain only as its CPS is disabled). All files `*_ctrl.py` files are meant to be run using:

* `cosmic-bsub-submit u-ak543_convert_ctrl.py`
    * runs all jobs defined in `u-ak543_convert_ctrl.py`

These will run the file defined by e.g. `SCRIPT_PATH = '/home/users/mmuetz/projects/cosmic/cosmic/processing/convert_pp_to_nc.py'`.



