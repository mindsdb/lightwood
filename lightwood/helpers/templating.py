from lightwood.api.dtype import dtype
'''
def is_allowed(v):
    if v is None:
        return True

    if isinstance(v, bool):
        return True

    try:
        float(v)
        return True
    except:
        pass

    if v in ['True', 'False']:
        return True

    if isinstance(v, str):
        if v.startswith('"') and v.endswith('"'):
            return True
        if v.startswith("'") and v.endswith("'"):
            return True

    # Predictor member
    if v.startswith('self.') and '(' not in v and len(v) < 50:
        return True

    # Allowed variable names
    if v in ['df', 'data', 'encoded_data', 'train_data', 'encoded_train_data', 'test_data']:
        return True

    try:
        cv = dict(v)
        for k in cv:
            ka = is_allowed(k)
            ma = is_allowed(cv[k])
            if not ka or not ma:
                return False
        return True
    except Exception:
        pass

    try:
        cv = list(v)
        for ma in cv:
            ma = is_allowed(m)
            if not ma:
                return False
        return True
    except Exception:
        pass

    raise Exception(f'Possible code injection: {v}')
'''


def is_allowed(v):
    if '(' in str(v):
        return False
    if 'lambda' in str(v):
        return False
    if '__' in str(v):
        return False

    return True


def call(entity: dict) -> str:
    # Special behavior for ensemble
    if 'submodels' in entity['args']:
        del entity['args']['submodels']

    for k, v in entity['args'].items():
        if not str(v).startswith('$'):
            if not is_allowed(v):
                raise Exception(f'Invalid value: {v} for arg {k}')

    args = [f'{k}={v}' for k, v in entity['args'].items() if not str(v).startswith('$')]

    for k, v in entity['args'].items():
        if str(v).startswith('$'):
            v = str(v).replace('$', 'self.')
            args.append(f'{k}={v}')

    args = ','.join(args)
    return f"""{entity['module']}({args})"""


def inline_dict(obj: dict) -> str:
    arr = []
    for k, v in obj.items():
        if str(v) in list(dtype.__dict__.keys()):
            v = f"'{v}'"
        k = k.replace("'", "\\'").replace('"', '\\"')
        arr.append(f"""'{k}': {v}""")

    dict_code = '{\n' + ',\n'.join(arr) + '\n}'
    return dict_code


def align(code: str, indent: int) -> str:
    add_space = ''
    for _ in range(indent):
        add_space += '    '

    code_arr = code.split('\n')
    code = f'\n{add_space}'.join(code_arr)
    return code
