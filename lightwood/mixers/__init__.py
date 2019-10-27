from lightwood.mixers.nn.nn import NnMixer
from lightwood.mixers.sk_learn.sk_learn import SkLearnMixer
from lightwood.mixers.bayesian_nn.bayesian_nn import BayesianNnMixer

class BuiltinMixers():
    NnMixer = NnMixer
    SkLearnMixer = SkLearnMixer
    BayesianNnMixer = BayesianNnMixer

BUILTIN_MIXERS = BuiltinMixers
