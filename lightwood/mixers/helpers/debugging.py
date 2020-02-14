import visdom
import subprocess


class TrainingMonitor():
    def __init__(sefl):
        self.vis = visdom.Visdom()

        self.train_loss_step = 0
        self.train_loss_layout = dict(title="Training Loss", xaxis={'title': 'loss'}, yaxis={'title': 'step'})

    def send_training_loss(loss, step):
        trace = dict(x=[loss], y=[step], mode="markers+lines", type='custom', marker={'color': 'red', 'symbol': 104, 'size': "10"})
        self.vis._send({'data': [trace], 'layout': self.train_loss_layout, 'win': 'mywin'})

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
