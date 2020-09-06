from lightwood.mixers.nn.nn import NnMixer
from lightwood.mixers.sk_learn.sk_learn import SkLearnMixer
try:
    from lightwood.mixers.boost.boost import BoostMixer
except ImportError:
    pass

from lightwood.mixers.base_mixer import BaseMixer


class BuiltinMixers():
    NnMixer = NnMixer
    SkLearnMixer = SkLearnMixer
    try:
        BoostMixer = BoostMixer
    except NameError:
        pass
      
BUILTIN_MIXERS = BuiltinMixers
