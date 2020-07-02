from hashlib import sha1


def task_unique_name_from_fn_args_kwargs(fn, args, kwargs):
    task_str = (fn.__code__.co_name +
                ''.join(str(a) for a in args) +
                ''.join(str(k) + str(v) for k, v in kwargs.items()))
    task_unique_filename = sha1(task_str.encode()).hexdigest() + '.task'
    return task_unique_filename


