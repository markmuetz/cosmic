import sys

from cosmic.util import load_module


if __name__ == '__main__':
    config = load_module(sys.argv[1])
    config_key = sys.argv[2]
    task = config.SCRIPT_ARGS[config_key]
    task.run()
