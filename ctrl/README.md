# ctrl

Control files for COSMIC. Broadly speaking these are files that are meant to be run. They might be run directly, using one of the commands defined in cosmic/bin/ or using remake. E.g.:

* `python orog_fracs.py`
* `cosmic-retrieve-from-mass u-ak543_ap9_precip.py`
    * retrieves all files defined in `u-ak543_ap9_precip.py`
* `cosmic-bsub-submit u-ak543_convert_ctrl.py`
    * runs all jobs defined in `u-ak543_convert_ctrl.py`
* `cosmic-bsub-task-submit plot_dem.py`
    * runs all tasks as bsub jobs defined in `plot_dem.py`
* `remake plot_dem.py`
    * runs all tasks defined in `plot_dem.py` (useful for testing)

