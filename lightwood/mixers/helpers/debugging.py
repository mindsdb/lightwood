import subprocess


def get_gpu_memory_map():
    '''
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    '''
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map

def print_gpuutil_status():
    import GPUtil
    GPUtil.showUtilization()
