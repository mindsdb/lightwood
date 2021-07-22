from lightwood.api.types import JsonAI


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
    if v in ['data', 'nfolds', 'data', 'train_data', 'test_data', 'df']:
        return True

    if isinstance(v, dict):
        for k in v:
            ka = is_allowed(k)
            ma = is_allowed(v[k])
            if not ka or not ma:
                return False
        return True

    if isinstance(v, list):
        for m in v:
            ma = is_allowed(m)
            if not ma:
                return False
        return True

    raise Exception(f'Possible code injection: {v}')
    

def call(entity: dict, json_ai: JsonAI) -> str:
    args = [f'{k}={v}' for k, v in entity['args'].items() if not str(v).startswith('$') and is_allowed(v)]

    for k, v in entity['args'].items():
        if str(v).startswith('$'):
            v = str(v).lstrip('$')
            try:
                val = json_ai
                for item in v.split('.'):
                    val = val.__getattribute__(item)
                    if isinstance(val, str):
                        val = f'"{val}"'
                args.append(f'{k}={val}')
            except Exception:
                args.append(f'{k}=self.{val}')

    return f"""{entity['module']}({args})"""


def inline_dict(obj: dict) -> str:
    arr = []
    for k, v in obj.items():
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