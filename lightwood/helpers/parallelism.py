import os
from typing import Dict
import psutil
import multiprocessing as mp


def get_nr_procs(df=None):
    if os.name == 'nt':
        return 1
    else:
        available_mem = psutil.virtual_memory().available
        try:
            import mindsdb_worker # noqa
            max_per_proc_usage = 0.2 * pow(10, 9)
        except Exception:
            max_per_proc_usage = 0.5 * pow(10, 9)

        if df is not None:
            max_per_proc_usage += df.memory_usage(index=True, deep=True).sum()
        proc_count = int(min(mp.cpu_count(), available_mem // max_per_proc_usage)) - 1

        return max(proc_count, 1)


def run_mut_method(obj: object, arg: object, method: str, identifier: str, return_dict: dict) -> str:
    try:
        obj.__getattribute__(method)(arg)
        return_dict[identifier] = obj
    except Exception:
        return_dict[identifier] = False


def mut_method_call(object_dict: Dict[str, tuple]) -> Dict[str, object]:
    manager = mp.Manager()
    return_dict = manager.dict()

    jobs = []
    for name, data in object_dict.items():
        p = mp.Process(target=run_mut_method, args=(data[0], data[1], data[2], name, return_dict))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    for identifier in return_dict:
        if return_dict[identifier] == False:
            raise Exception(f'Failed to run in parallel on identifier: {identifier}')

    return dict(return_dict)