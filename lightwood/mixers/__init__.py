from lightwood.mixers.base_mixer import BaseMixer

try:
    from lightwood.mixers.nn import NnMixer
except Exception as e:
    print(e)
    NnMixer = None

try:
    from lightwood.mixers.lightgbm import LightGBMMixer
except Exception as e:
    print(e)
    LightGBMMixer = None

# @TODO: Bellow mixers not ready for prime time yet
# BoostMixer should probably go away in favor of LightGBMMixer
# SklearnMixer has potential to be an auto-sklearn like thing, but nobody has the time to work on it
# On hold til we have better benchmarks and more time
try:
    from lightwood.mixers.boost import BoostMixer
except Exception as e:
    BoostMixer = None

try:
    from lightwood.mixers.sklearn_mixer import SklearnMixer
except Exception as e:
    SklearnMixer = None
