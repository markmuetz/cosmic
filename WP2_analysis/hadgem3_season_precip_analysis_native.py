import sys

from cosmic.util import load_config


if __name__ == '__main__':
    config = load_config(sys.argv[1])
    config_key = sys.argv[2]
    task = config.SCRIPT_ARGS[config_key]
    task.run()
