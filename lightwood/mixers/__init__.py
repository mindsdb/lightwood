from lightwood.mixers.nn.nn import NnMixer
from lightwood.mixers.sk_learn.sk_learn import SkLearnMixer
from lightwood.mixers.boost.boost import BoostMixer

class BuiltinMixers():
    NnMixer = NnMixer
    SkLearnMixer = SkLearnMixer
    BoostMixer = BoostMixer

BUILTIN_MIXERS = BuiltinMixers
