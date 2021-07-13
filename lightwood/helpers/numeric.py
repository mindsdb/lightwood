'''
To: future intern reading this bit of code thinking "this is dumb, we could just check for isnan and isinf

No, no you can't, this will break subtly in a bunch of edge cases, see the example (+ comments) in this article: https://cerebralab.com/Exceptions_as_control_flow

Sorry, python is dumb, but the spirit of your investigation wasn't, try again
'''


def can_be_nan_numeric(value):
    try:
        value = str(value)
        value = float(value)
    except Exception:
        return False

    try:
        if isinstance(value, float):
            a = int(value)
        isnan = False
    except Exception:
        isnan = True
    return isnan


def filter_nan(series):
    return [x for x in series if not can_be_nan_numeric(x)]
