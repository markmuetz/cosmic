import sys
from hashlib import sha1
from pathlib import Path

from cosmic.util import load_config


def main(config_filename, task_path_hash_key, config_path_hash):
    config_path = Path(config_filename).absolute()
    curr_config_path_hash = sha1(config_path.read_bytes()).hexdigest()
    if config_path_hash != curr_config_path_hash:
        raise Exception(f'config file {config_path} has changed -- cannot run task.')

    config = load_config(config_filename)
    task_ctrl = config.task_ctrl
    if not task_ctrl.finalized:
        task_ctrl.finalize()
    task = task_ctrl.task_from_path_hash_key[task_path_hash_key]
    task.run()


if __name__ == '__main__':
    print(sys.argv)
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])
