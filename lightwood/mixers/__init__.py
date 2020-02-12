from lightwood.mixers.nn.nn import NnMixer
from lightwood.mixers.sk_learn.sk_learn import SkLearnMixer
try:
    from lightwood.mixers.boost.boost import BoostMixer
except:
    pass


class BuiltinMixers():
    NnMixer = NnMixer
    SkLearnMixer = SkLearnMixer
    try:
        BoostMixer = BoostMixer
    except:
        pass
      
BUILTIN_MIXERS = BuiltinMixers
