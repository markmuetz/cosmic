import sys

from cosmic.util import load_config


def main(config_filename, task_index):
    config = load_config(config_filename)
    task_ctrl = config.task_ctrl
    if not task_ctrl.finalized:
        task_ctrl.finalize()
    task = task_ctrl.task_run_schedule[task_index]
    task.run()


if __name__ == '__main__':
    print(sys.argv)
    main(sys.argv[1], int(sys.argv[2]))
