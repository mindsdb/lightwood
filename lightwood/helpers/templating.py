from lightwood.api.types import JsonAI


def is_allowed(v):
    for t in [bool, str]:
        if isinstance(v, t):
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

    return False

def call(entity: dict, json_ai: JsonAI) -> str:
    dynamic_args = [f'{k}={v}' for k, v in entity['dynamic_args'].items() if not str(v).startswith('$') and is_allowed(v)]

    static_args = []
    for k, v in entity['dynamic_args'].items():
        if str(v).startswith('$'):
            v = str(v).lstrip('$')
            val = json_ai
            for item in v.split('.'):
                val = val.__getattribute__(item)
                if isinstance(val, str):
                    val = f'"{val}"'
            static_args.append(f'{k}={val}')

    args = ', '.join(static_args + dynamic_args)

    call = entity['object']

    return f'{call}({args})'


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