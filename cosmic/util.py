from pathlib import Path
import importlib.util
import subprocess as sp

from cosmic.cosmic_errors import CosmicError


def load_config(local_filename):
    config_filepath = Path.cwd() / local_filename
    if not config_filepath.exists():
        raise CosmicError(f'Config file {config_filepath} does not exist')

    try:
        spec = importlib.util.spec_from_file_location('config', config_filepath)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
    except SyntaxError as se:
        print(f'Bad syntax in config file {config_filepath}')
        raise

    return config


def sysrun(cmd):
    """Run a system command, returns a CompletedProcess

    raises CalledProcessError if cmd is bad.
    to access output: sysrun(cmd).stdout"""
    return sp.run(cmd, check=True, shell=True, stdout=sp.PIPE, stderr=sp.PIPE, encoding='utf8') 

