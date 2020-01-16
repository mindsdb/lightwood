import runpy

pdir = '../../lightwood/'
encoders_path = pdir + 'encoders/'
mixers_path = pdir + 'mixers/'
MODULES = [
    f'{encoders_path}categorical/onehot.py',
    f'{encoders_path}datetime/datetime.py',
    f'{mixers_path}helpers/plinear.py',
    f'{pdir}api/data_source.py',
    f'{mixers_path}nn/nn.py',
    f'{encoders_path}categorical/autoencoder.py',
    #f'{encoders_path}text/rnn.py'.format(encoders_path),
    #'./lightwood/encoders/image/nn.py',
    #'./lightwood/encoders/image/img_2_vec.py',
    #'./lightwood/encoders/numeric/numeric.py',
    #'./lightwood/encoders/time_series/cesium_ts.py',
    #'./lightwood/mixers/sk_learn/sk_learn.py',
    #'./lightwood/encoders/text/infersent.py',
    #'./lightwood/encoders/text/distilbert.py'
]


def run_tests(modules):
    '''
    Run modules as scripts to execute main function
    '''    
    for module in modules:
        runpy.run_path(module, run_name='__main__')


if __name__ == "__main__":
    run_tests(MODULES)