import os
import logging
import subprocess as sp
from pathlib import Path
from timeit import default_timer as timer

from cosmic.util import load_config, sysrun


BSUB_SCRIPT_TPL = """#!/bin/bash
#BSUB -J {job_name}
#BSUB -q {queue}
#BSUB -o processing_output/{script_name}_{config_name}_{config_key}_%J.out
#BSUB -e processing_output/{script_name}_{config_name}_{config_key}_%J.err
#BSUB -W {max_runtime}

python {script_path} {config_path} {config_key}
"""


logging.basicConfig(level=os.getenv('COSMIC_LOGLEVEL', 'INFO'), 
                    format='%(asctime)s %(levelname)8s: %(message)s')
logger = logging.getLogger(__name__)


def write_bsub_script(bsub_dir, script_path, config_path, config_key, bsub_kwargs):
    script_name = script_path.stem
    config_name = config_path.stem

    bsub_script_filepath = bsub_dir / f'{script_name}_{config_name}_{config_key}.bsub'
    logger.debug(f'  writing {bsub_script_filepath}')

    bsub_script = BSUB_SCRIPT_TPL.format(script_name=script_name,
                                         script_path=script_path,
                                         config_name=config_name,
                                         config_path=config_path,
                                         config_key=config_key,
                                         **bsub_kwargs)

    with open(bsub_script_filepath, 'w') as fp:
        fp.write(bsub_script)
    return bsub_script_filepath


def submit_bsub_script(bsub_script_path):
    try:
        comp_proc = sysrun(f'bsub < {bsub_script_path}')
        logger.info(comp_proc.stdout)
    except sp.CalledProcessError as cpe:
        logger.error(f'Error submitting {bsub_script_path}')
        logger.error(cpe)
        logger.error('===ERROR===')
        logger.error(cpe.stderr)
        logger.error('===ERROR===')
        raise


def main(config_filename):
    config = load_config(config_filename)
    logger.debug(config)

    bsub_dir = Path('bsub_scripts')
    bsub_dir.mkdir(exist_ok=True)
    output_dir = Path('processing_output')
    output_dir.mkdir(exist_ok=True)

    script_path = Path(config.SCRIPT_PATH).absolute()
    config_path = Path(config_filename).absolute()
    logger.debug(script_path)
    logger.debug(config_path)

    for config_key in config.CONFIG_KEYS:
        bsub_script_filepath = write_bsub_script(bsub_dir, script_path, config_path, 
                                                 config_key, config.BSUB_KWARGS)
        submit_bsub_script(bsub_script_filepath)
