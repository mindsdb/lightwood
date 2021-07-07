import os
from typing import Dict
import psutil
import multiprocessing as mp
from lightwood.helpers.log import log


def get_nr_procs(df=None):
    if os.name == 'nt':
        return 1
    else:
        available_mem = psutil.virtual_memory().available
        max_per_proc_usage = 0.2 * pow(10, 9)

        if df is not None:
            max_per_proc_usage += df.memory_usage(index=True, deep=True).sum()
        proc_count = int(min(mp.cpu_count(), available_mem // max_per_proc_usage)) - 1

        return max(proc_count, 1)


def run_mut_method(obj: object, arg: object, method: str, identifier: str, return_dict: dict) -> str:
    try:
        obj.__getattribute__(method)(arg)
        log.info(f'Got a result, return dict lenght now at {len(return_dict)}')
        return_dict[identifier] = obj
    except Exception as e:
        return_dict[identifier] = False
        raise e


def mut_method_call(object_dict: Dict[str, tuple]) -> Dict[str, object]:
    manager = mp.Manager()
    return_dict = manager.dict()

    nr_procs = get_nr_procs()
    pool = mp.Pool(processes=nr_procs)
    for name, data in object_dict.items():
        pool.apply_async(func=run_mut_method, args=(data[0], data[1], data[2], name, return_dict))
    pool.close()
    pool.join()

    for identifier in return_dict:
        if return_dict[identifier] == False:
            raise Exception(f'Failed to run in parallel on identifier: {identifier}')

    return dict(return_dict)