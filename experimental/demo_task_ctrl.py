from cosmic.task import Task, TaskControl

BSUB_KWARGS = {
    'job_name': 'demo',
    'queue': 'short-serial',
    'max_runtime': '00:10',
    # 'mem': '64000',
}


def task_fn(inputs, outputs):
    print(inputs)
    print(outputs)
    for o in outputs:
        o.touch()


task_ctrl = TaskControl()
task_ctrl.add(Task(task_fn, [], ['task_demo1.out']))
task_ctrl.add(Task(task_fn, [], ['task_demo2.out']))
task_ctrl.add(Task(task_fn, [], ['task_demo3.out']))
task_ctrl.add(Task(task_fn, ['task_demo1.out', 'task_demo2.out', 'task_demo3.out'], ['task_demo4.out']))
task_ctrl.add(Task(task_fn, ['task_demo4.out'], ['task_demo5.out']))
task_ctrl.add(Task(task_fn, ['task_demo1.out', 'task_demo5.out'], ['task_demo6.out']))
