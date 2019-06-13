
from lightwood.mixers.nn.nn import NnMixer
from lightwood.mixers.sk_learn.sk_learn import SkLearnMixer

class BuiltinMixers():
    NnMixer = NnMixer
    SkLearnMixer = SkLearnMixer

BUILTIN_MIXERS = BuiltinMixers