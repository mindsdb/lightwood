from sklearn.metrics import r2_score as sk_r2_score


def r2_score(y_true, y_pred) -> float:
    """ Wrapper for sklearn R2 score, lower capped at 0. """
    for arr in (y_true, y_pred):
        for i in range(len(arr)):
            try:
                if arr[i] is None or np.isnan(arr[i]):
                    arr[i] = 0
                if np.isinf(arr[i]):
                    arr[i] = pow(2, 63)
            except Exception as e:
                print(f'Strange value {arr[i]} caused exception: {e}')
                arr[i] = 0
    acc = sk_r2_score(y_true, y_pred)
    return min(1, max(0, acc))
