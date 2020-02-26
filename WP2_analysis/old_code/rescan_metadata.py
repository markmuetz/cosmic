# coding: utf-8
from basin_weighted_diurnal_cycle_analysis import gen_task_ctrl
from pathlib import Path
from remake.metadata import PathMetadata, TaskMetadata

task_ctrl = gen_task_ctrl()
task_ctrl.finalize()

from remake.setup_logging import setup_stdout_logging; setup_stdout_logging('DEBUG')
# task_ctrl.rescan_metadata()
