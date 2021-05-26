import os
import psutil
import multiprocessing as mp


def get_nr_procs(df=None):
    if os.name == 'nt':
        return 1
    else:
        available_mem = psutil.virtual_memory().available
        try:
            import mindsdb_worker
            max_per_proc_usage = 0.2 * pow(10,9)
        except:
            max_per_proc_usage = 0.5 * pow(10, 9)

        if df is not None:
            max_per_proc_usage += df.memory_usage(index=True, deep=True).sum()
        proc_count = int(min(mp.cpu_count(), available_mem // max_per_proc_usage)) - 1

        return max(proc_count, 1)
