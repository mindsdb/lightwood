import subprocess

import visdom
import numpy as np


class TrainingMonitor():
    def __init__(self, name='Default'):
        self.vis = visdom.Visdom()
        self.name = name

        self.loss_wds = {}
        self.loss_colors = [[211,47,47],[81,45,168],[2,136,209],[56,142,60],[251,192,45],[230,74,25],[69,90,100]]
        self.loss_colors_index = 0


    def weight_map(self, layer_names, values, name):
        max_len = 0
        min_val = 0

        for array in values:
            max_len = max(len(array), max_len)
            min_val = min(min(array), min_val)

        for array in values:
            nr_to_append = max_len - len(array)
            array.extend([min_val] * nr_to_append)

        self.vis.heatmap(
            X=values,
            opts=dict(
                rownames=layer_names,
                colormap='Electric',
                title=name,
            )
            ,win=name
        )

    def plot_loss(self, loss, step, name):
        if name not in self.loss_wds:
            self.loss_wds[name] = {
                'name': f'{self.name} - {name}'
                ,'color': self.loss_colors[self.loss_colors_index]
            }
            self.loss_colors_index += 1
            if self.loss_colors_index >= len(self.loss_colors):
                self.loss_colors_index = 0

            self.vis.line(X=[step],Y=[loss],win=self.loss_wds[name]['name'],
                opts=dict(
                    width=600,
                    height=600,
                    xlabel='Step',
                    ylabel='Loss',
                    title=self.loss_wds[name]['name'],
                    linecolor=np.array([self.loss_wds[name]['color']])
                )
            )
        else:
            self.vis.line(X=[step],Y=[loss],win=self.loss_wds[name]['name'],update='append')

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
