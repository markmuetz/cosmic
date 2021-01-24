import os
import re
import sys
from argparse import ArgumentParser
from hashlib import sha1
import json
import logging
import subprocess as sp
from pathlib import Path

from cosmic.util import sysrun
import cosmic.processing.bsub_task_run as bsub_task_run

from remake.setup_logging import add_file_logging
from remake.task_control import load_task_ctrls

SLURM_SCRIPT_TPL = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH -p {queue}
#SBATCH -o processing_output/{script_name}_{config_name}_{task_path_hash_key}_%j.out
#SBATCH -e processing_output/{script_name}_{config_name}_{task_path_hash_key}_%j.err
#SBATCH --time={max_runtime}
#SBATCH --mem={mem}
{dependencies}

python {script_path} {config_path} {task_path_hash_key} {config_path_hash}
"""


logging.basicConfig(stream=sys.stdout, level=os.getenv('COSMIC_LOGLEVEL', 'INFO'),
                    format='%(asctime)s %(levelname)8s: %(message)s')
logger = logging.getLogger(__name__)


def _parse_jobid(output):
    if match := re.match('Submitted batch job (?P<jobid>\d+)', output):
        jobid = match['jobid']
        return jobid
    else:
        raise Exception(f'Could not parse {output}')


def _submit_slurm_script(slurm_script_path):
    try:
        comp_proc = sysrun(f'sbatch {slurm_script_path}')
        output = comp_proc.stdout
        logger.info(output)
    except sp.CalledProcessError as cpe:
        logger.error(f'Error submitting {slurm_script_path}')
        logger.error(cpe)
        logger.error('===ERROR===')
        logger.error(cpe.stderr)
        logger.error('===ERROR===')
        raise
    return output


class TaskSubmitter:
    def __init__(self, slurm_dir, config_path, task_ctrl, slurm_kwargs):
        self.slurm_dir = slurm_dir
        self.config_path = config_path
        self.task_ctrl = task_ctrl
        self.slurm_kwargs = slurm_kwargs
        self.task_jobid_map = {}
        self.config_path_hash = sha1(config_path.read_bytes()).hexdigest()

    def _write_submit_script(self, task):
        config_name = self.config_path.stem
        script_path = Path(bsub_task_run.__file__)
        script_name = script_path.stem
        slurm_script_filepath = self.slurm_dir / f'{script_name}_{config_name}_{task.path_hash_key()}.sbatch'
        logger.debug(f'  writing {slurm_script_filepath}')
        if 'mem' not in self.slurm_kwargs:
            self.slurm_kwargs['mem'] = 16000

        prev_jobids = []
        prev_tasks = self.task_ctrl.prev_tasks[task]
        for prev_task in prev_tasks:
            # N.B. not all dependencies have to have been run; they could not require rerunning.
            if prev_task in self.task_jobid_map:
                prev_jobids.append(self.task_jobid_map[prev_task])
        if prev_jobids:
            dependencies = '#SBATCH --dependency=afterok:' + ':'.join(prev_jobids)
        else:
            dependencies = ''

        slurm_script = SLURM_SCRIPT_TPL.format(script_name=script_name,
                                               script_path=script_path,
                                               config_name=config_name,
                                               config_path=self.config_path,
                                               task_path_hash_key=task.path_hash_key(),
                                               dependencies=dependencies,
                                               config_path_hash=self.config_path_hash,
                                               job_name=task.path_hash_key()[:10],  # Any longer and a leading * is added.
                                               **self.slurm_kwargs)

        with open(slurm_script_filepath, 'w') as fp:
            fp.write(slurm_script)
        return slurm_script_filepath

    def submit_task(self, task):
        slurm_script_path = self._write_submit_script(task)
        output = _submit_slurm_script(slurm_script_path)
        jobid = _parse_jobid(output)
        self.task_jobid_map[task] = jobid


def main():
    parser = ArgumentParser()
    parser.add_argument('--config-filename', '-C')
    parser.add_argument('--queue', '-q', default='short-serial')
    parser.add_argument('--max-runtime', '-W', default='04:00:00')
    parser.add_argument('--mem', '-M', default=16000)
    parser.add_argument('--ntasks', '-N', type=int, default=int(1e9))
    args = parser.parse_args()

    slurm_kwargs = {'queue': args.queue,
                   'max_runtime': args.max_runtime,
                   'mem': args.mem}

    output_dir = Path('processing_output')
    output_dir.mkdir(exist_ok=True)
    add_file_logging(output_dir / 'slurm_task_submit.log')

    config_path = Path(args.config_filename).absolute()
    task_ctrl = load_task_ctrls(config_path)[0]

    slurm_dir = Path('slurm_scripts')
    slurm_dir.mkdir(exist_ok=True)

    task_ctrl.finalize()

    submitter = TaskSubmitter(slurm_dir, config_path, task_ctrl, slurm_kwargs)

    tasks_to_submit = []
    for task in task_ctrl.sorted_tasks:
        if task in task_ctrl.pending_tasks or task in task_ctrl.remaining_tasks:
            tasks_to_submit.append(task)
            if len(tasks_to_submit) >= args.ntasks:
                break

    for i, task in enumerate(tasks_to_submit):
        logger.info(f'task {i + 1}/{len(tasks_to_submit)}: {task}')
        submitter.submit_task(task)

    Path('processing_output/submitted_tasks.json').write_text(json.dumps([(t.path_hash_key(), repr(t))
                                                                         for t in tasks_to_submit]))


