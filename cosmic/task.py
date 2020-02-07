from _collections import defaultdict
from pathlib import Path
import pickle

import iris
import pandas as pd


LOADERS = {
    '.nc': lambda fn: iris.load(str(fn)),
    '.hdf': lambda fn: pd.read_hdf(fn, fn.stem),
    '.pkl': lambda fn: pickle.load(fn.open('rb')),
}
SAVERS = {
    '.nc': lambda obj, fn: iris.save(obj, str(fn)),
    '.hdf': lambda obj, fn: obj.to_hdf(fn, fn.stem),
    '.pkl': lambda obj, fn: pickle.dump(obj, fn.open('wb')),
}


class TaskControl:
    def __init__(self):
        self.tasks = []
        self.task_output_map = {}
        self.input_task_map = defaultdict(list)
        self.finilized = False
        self.index = None
        self.task_run_schedule = []

    def _filter_tasks(self, fn):
        if fn:
            tasks = [t for t in self.tasks if t.fn == fn]
        else:
            tasks = self.tasks
        return tasks

    def _gen_index(self, fn=None):
        tasks = self._filter_tasks(fn)
        columns = set()
        for task in tasks:
            columns |= set(task.kwargs.keys())
        columns = sorted(columns)
        data = []
        for task in tasks:
            data.append([task.kwargs.get(c) for c in columns] + [task])
        return pd.DataFrame(data, columns=columns + ['task'])

    def add(self, task):
        if self.finilized:
            raise Exception(f'TaskControl already finilized')

        self.tasks.append(task)
        for output in task.outputs:
            if output in self.task_output_map:
                raise Exception(f'Trying to add {output} twice')
            self.task_output_map[output] = task
            for input_fn in task.inputs:
                self.input_task_map[input_fn].append(task)
        return task

    def finilize(self):
        assert not self.finilized
        task_run_schedule = []
        tasks = [t for t in self.tasks]
        can_run_tasks = set()

        # Work out whether it is possible to create a run schedule and find initial tasks.
        for task in tasks:
            for input_fn in task.inputs:
                if input_fn not in self.task_output_map:
                    if not input_fn.exists():
                        raise Exception(f'No input files exist or will be created for {task}')
                    self.task_run_schedule.append(task)
                    can_run_tasks.add(task)
                    tasks.remove(task)

        # It is possible; build remainder of schedule.
        while tasks:
            for task in tasks:
                can_run = True
                for input_fn in task.inputs:
                    intask = self.task_output_map[input_fn]
                    if intask not in can_run_tasks:
                        can_run = False
                if can_run:
                    task_run_schedule.append(task)
                    can_run_tasks.add(task)
                    tasks.remove(task)
        self.task_run_schedule = task_run_schedule

        self.index = self._gen_index()
        self.finilized = True
        return self

    def get_deps(self, task, depth=1):
        assert self.finilized
        level = 1
        deps = defaultdict(set)
        level_tasks = {task}
        next_level_tasks = set()
        while depth:
            for level_task in level_tasks:
                for input_fn in level_task.inputs:
                    next_level_task = self.task_output_map[input_fn]
                    next_level_tasks.add(next_level_task)
                    deps[level].add(next_level_task)
            level_tasks = set(next_level_tasks)
            next_level_tasks = set()

            depth -= 1
            level += 1
        return deps

    def run(self, fn=None, force=False):
        assert self.finilized
        if fn:
            tasks = [t for t in self.task_run_schedule if t.fn == fn]
        else:
            tasks = [t for t in self.task_run_schedule]
        for i, task in enumerate(tasks):
            print(f'{i + 1}/{len(tasks)}: {task.fn.__code__.co_name} - {task.outputs}')
            task.run(force=force)


class Task:
    def __init__(self, fn, inputs, outputs,
                 fn_args=[], fn_kwargs={},
                 save_output=False,
                 **kwargs):
        self.fn = fn
        self.fn_args = fn_args
        self.fn_kwargs = fn_kwargs
        if not outputs:
            raise Exception('outputs must be set')

        self.inputs = [Path(i) for i in inputs]
        self.outputs = [Path(o) for o in outputs]
        self.save_output = save_output
        self.kwargs = kwargs
        self.result = None

        self.rerun_on_mtime = True

    def __repr__(self):
        return f'Task({self.fn.__code__.co_name}, {self.inputs}, {self.outputs})'

    def can_run(self):
        can_run = True
        for input_fn in self.inputs:
            if not input_fn.exists():
                can_run = False
                break
        return can_run

    def requires_rerun(self):
        rerun = False
        earliest_output_path_mtime = float('inf')
        for output in self.outputs:
            if not Path(output).exists():
                rerun = True
                break
            earliest_output_path_mtime = min(earliest_output_path_mtime,
                                             output.stat().st_mtime)
        if self.rerun_on_mtime and not rerun:
            latest_input_path_mtime = 0
            for input_fn in self.inputs:
                latest_input_path_mtime = max(latest_input_path_mtime,
                                              input_fn.stat().st_mtime)
            if latest_input_path_mtime > earliest_output_path_mtime:
                rerun = True

        return rerun

    def load_outputs(self, *args, **kwargs):
        if self.result:
            return self.result
        loaded_outputs = []
        for output in self.outputs:
            if output.suffix in LOADERS:
                loaded_outputs.append(LOADERS[output.suffix](output, *args, **kwargs))
            else:
                raise Exception(f'Unknown filetype: {output}')
        return loaded_outputs

    def load_output(self, *args, **kwargs):
        outputs = self.load_outputs(*args, **kwargs)
        assert len(outputs) == 1
        return outputs[0]

    def run(self, force=False):
        if not self.can_run():
            raise Exception('Not all files required for task exist')

        if self.requires_rerun() or force:
            self.result = self.fn(self.inputs, self.outputs, *self.fn_args, **self.fn_kwargs)
            if self.save_output:
                assert len(self.result) == len(self.outputs)
                for r, o in zip(self.result, self.outputs):
                    SAVERS[o.suffix](r, o)

            for output in self.outputs:
                if not output.exists():
                    raise Exception(f'fn {output} not created')
        else:
            print(f'  Already exist: {self.outputs}')

        return self

