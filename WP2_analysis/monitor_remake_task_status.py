import sys
from collections import Counter
import datetime as dt
from pathlib import Path
from time import sleep

import curses
from curses import wrapper

# from basin_weights_analysis import gen_task_ctrl

def main(stdscr, status_dir='.remake/metadata_v3/task_status'):
    status_dir = Path(status_dir)

    # task_ctrl = gen_task_ctrl(False)
    curses.curs_set(0)
    while True:
        stdscr.clear()

        paths = status_dir.glob('*.status')
        statuses = Counter()
        running = []
        for path in paths:
            time, status = path.read_text().split('\n')[-2].split(';')
            statuses[status] += 1
            if status == 'RUNNING':
                running.append(path)
        stdscr.addstr(0, 0, f'Time    : {dt.datetime.now()}')
        stdscr.addstr(1, 0, f'Complete: {statuses["COMPLETE"]}')
        stdscr.addstr(2, 0, f'Running : {statuses["RUNNING"]}')
        stdscr.addstr(3, 0, f'Error   : {statuses["ERROR"]}')
        for i, path in enumerate(running):
            stdscr.addstr(5 + i, 0, f'{path.stem}')

        stdscr.refresh()
        curses.napms(1000)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        wrapper(main, sys.argv[1])
    else:
        wrapper(main)
