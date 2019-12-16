import runpy

MODULES = [
    './lightwood/encoders/categorical/onehot.py',
    './lightwood/encoders/datetime/datetime.py',
    #'./lightwood/encoders/image/nn.py',
    #'./lightwood/encoders/image/img_2_vec.py',
    './lightwood/encoders/text/rnn.py',
    './lightwood/encoders/numeric/numeric.py',
    './lightwood/mixers/helpers/plinear.py',
    #'./lightwood/encoders/categorical/autoencoder.py',
    './lightwood/encoders/time_series/cesium_ts.py',
   # './lightwood/mixers/sk_learn/sk_learn.py',
    './lightwood/mixers/nn/nn.py',
    #'./lightwood/encoders/text/infersent.py',
    './lightwood/api/data_source.py'
]

def run_tests(modules):
    
    for module in modules:
        runpy.run_path(module, run_name='__main__')


if __name__ == "__main__":
    run_tests(MODULES)