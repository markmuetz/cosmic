import os
import sys
from argparse import ArgumentParser
from hashlib import sha1
import json
import logging
import subprocess as sp
from pathlib import Path

from cosmic.util import load_config, sysrun
import cosmic.processing.bsub_task_run as bsub_task_run


BSUB_SCRIPT_TPL = """#!/bin/bash
#BSUB -J {job_name}
#BSUB -q {queue}
#BSUB -o processing_output/{script_name}_{config_name}_{task_path_hash_key}_%J.out
#BSUB -e processing_output/{script_name}_{config_name}_{task_path_hash_key}_%J.err
#BSUB -W {max_runtime}
#BSUB -M {mem}
{dependencies}

python {script_path} {config_path} {task_path_hash_key} {config_path_hash}
"""


logging.basicConfig(stream=sys.stdout, level=os.getenv('COSMIC_LOGLEVEL', 'INFO'),
                    format='%(asctime)s %(levelname)8s: %(message)s')
logger = logging.getLogger(__name__)


def _parse_jobid(output):
    return output[output.find('<') + 1:output.find('>')]


def _submit_bsub_script(bsub_script_path):
    try:
        comp_proc = sysrun(f'bsub < {bsub_script_path}')
        output = comp_proc.stdout
        logger.info(output)
    except sp.CalledProcessError as cpe:
        logger.error(f'Error submitting {bsub_script_path}')
        logger.error(cpe)
        logger.error('===ERROR===')
        logger.error(cpe.stderr)
        logger.error('===ERROR===')
        raise
    return output


class TaskSubmitter:
    def __init__(self, bsub_dir, config_path, task_ctrl, bsub_kwargs):
        self.bsub_dir = bsub_dir
        self.config_path = config_path
        self.task_ctrl = task_ctrl
        self.bsub_kwargs = bsub_kwargs
        self.task_jobid_map = {}
        self.config_path_hash = sha1(config_path.read_bytes()).hexdigest()

    def _write_submit_script(self, task):
        config_name = self.config_path.stem
        script_path = Path(bsub_task_run.__file__)
        script_name = script_path.stem
        bsub_script_filepath = self.bsub_dir / f'{script_name}_{config_name}_{task.path_hash_key()}.bsub'
        logger.debug(f'  writing {bsub_script_filepath}')
        if 'mem' not in self.bsub_kwargs:
            self.bsub_kwargs['mem'] = 16000

        prev_jobids = []
        prev_tasks = self.task_ctrl.prev_tasks[task]
        for prev_task in prev_tasks:
            # N.B. not all dependencies have to have been run; they could not require rerunning.
            if prev_task in self.task_jobid_map:
                prev_jobids.append(self.task_jobid_map[prev_task])
        if prev_jobids:
            dependencies = '#BSUB -w "' + ' && '.join([f'done({jobid})' for jobid in prev_jobids]) + '"'
        else:
            dependencies = ''

        bsub_script = BSUB_SCRIPT_TPL.format(script_name=script_name,
                                             script_path=script_path,
                                             config_name=config_name,
                                             config_path=self.config_path,
                                             task_path_hash_key=task.path_hash_key(),
                                             dependencies=dependencies,
                                             config_path_hash=self.config_path_hash,
                                             **self.bsub_kwargs)

        with open(bsub_script_filepath, 'w') as fp:
            fp.write(bsub_script)
        return bsub_script_filepath

    def submit_task(self, task):
        bsub_script_path = self._write_submit_script(task)
        output = _submit_bsub_script(bsub_script_path)
        jobid = _parse_jobid(output)
        self.task_jobid_map[task] = jobid


def main():
    parser = ArgumentParser()
    parser.add_argument('--config-filename', '-C')
    parser.add_argument('--ntasks', '-N', type=int, default=int(1e9))
    args = parser.parse_args()

    config = load_config(args.config_filename)
    logger.debug(config)

    bsub_dir = Path('bsub_scripts')
    bsub_dir.mkdir(exist_ok=True)
    output_dir = Path('processing_output')
    output_dir.mkdir(exist_ok=True)

    config_path = Path(args.config_filename).absolute()
    logger.debug(config_path)

    task_ctrl = config.gen_task_ctrl()
    task_ctrl.enable_file_task_content_checks = False

    if not task_ctrl.finalized:
        task_ctrl.finalize()

    submitter = TaskSubmitter(bsub_dir, config_path, task_ctrl, config.BSUB_KWARGS)

    tasks_to_submit = []
    task_count = 0
    for task in task_ctrl.sorted_tasks:
        if task not in task_ctrl.pending_tasks and task not in task_ctrl.remaining_tasks:
            continue
        task_count += 1
        tasks_to_submit.append(task)
        if task_count >= args.ntasks:
            break

    for i, task in enumerate(tasks_to_submit):
        logger.info(f'task {i + 1}/{len(tasks_to_submit)}: {task}')
        submitter.submit_task(task)

    Path('processing_output/submitted_tasks.json').write_text(json.dumps([(t.hexdigest(), repr(t))
                                                                         for t in tasks_to_submit]))

