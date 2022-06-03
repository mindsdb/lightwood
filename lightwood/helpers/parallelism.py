import os
from typing import Dict
import psutil
import multiprocessing as mp
from lightwood.helpers.log import log


def get_nr_procs(df=None):
    if 'LIGHTWOOD_N_WORKERS' in os.environ:
        try:
            n = int(os.environ['LIGHTWOOD_N_WORKERS'])
        except ValueError:
            n = 1
        return n
    elif os.name == 'nt':
        return 1
    else:
        available_mem = psutil.virtual_memory().available
        max_per_proc_usage = 0.2 * pow(10, 9)

        if df is not None:
            max_per_proc_usage += df.memory_usage(index=True, deep=True).sum()
        proc_count = int(min(mp.cpu_count(), available_mem // max_per_proc_usage)) - 1

        return max(proc_count, 1)


def run_mut_method(obj: object, arg: object, method: str, identifier: str) -> str:
    try:
        obj.__getattribute__(method)(arg)
        return obj, identifier
    except Exception as e:
        log.error(f'Exception {e} when running with identifier {identifier}')
        raise e


def mut_method_call(object_dict: Dict[str, tuple]) -> Dict[str, object]:
    manager = mp.Manager()
    return_dict = manager.dict()

    nr_procs = get_nr_procs()
    pool = mp.Pool(processes=nr_procs)
    promise_arr = []
    for name, data in object_dict.items():
        promise = pool.apply_async(func=run_mut_method, args=(data[0], data[1], data[2], name))
        promise_arr.append(promise)

    for promise in promise_arr:
        obj, identifier = promise.get()
        return_dict[identifier] = obj
        log.info(f'Done running for: {identifier}')

    pool.close()
    pool.join()

    return dict(return_dict)
