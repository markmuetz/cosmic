import sys

from cosmic.util import load_config


def main(config_filename, task_index):
    config = load_config(config_filename)
    task_ctrl = config.task_ctrl
    task = task_ctrl.task_run_schedule[task_index]
    task.run()


if __name__ == '__main__':
    main(sys.argv[0], int(sys.argv[1]))
