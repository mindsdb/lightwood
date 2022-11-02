from collections import deque

import numpy as np

from type_infer.dtype import dtype


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


def _consolidate_analysis_blocks(jsonai, key):
    """
    Receives a list of analysis blocks (where applicable, already filed with `hidden` args) and modifies it so that:
        1. All dependencies are correct.
        2. Execution order is such that all dependencies are met.
            - For this we use a topological sort over the DAG.
    """
    analysis_defaults = {  # non-optional plus dependencies
        'ICP': {
            "deps": [],
        },
        'AccStats': {
            "deps": ['ICP']
        },
        'ConfStats': {
            "deps": ['ICP']
        },
        'PermutationFeatureImportance': {
            "deps": ['AccStats']
        },
        'ShapleyValues': {
            "deps": []
        },
        'TempScaler': {
            "deps": []
        }
    }

    # 1. all dependencies are correct
    blocks = getattr(jsonai, key)
    block_objs = {b['module']: b for b in blocks}

    for i, block in enumerate(blocks):
        if 'args' not in block:
            blocks[i]['args'] = analysis_defaults.get(block['module'], {"deps": []})
        elif 'deps' not in block['args']:
            blocks[i]['args']['deps'] = []
        for dep in block['args']['deps']:
            if dep not in block_objs.keys():
                raise Exception(f'Analysis block "{dep}" not found but necessary for block "{block["module"]}". Please add it and try again.')  # noqa

    # 2. correct execution order -- build a DAG out of analysis blocks
    block_objs = {b['module']: b for b in blocks}
    block_ids = {k: i for i, k in enumerate(block_objs.keys())}
    idx2block = {i: k for i, k in enumerate(block_objs.keys())}

    adj_M = np.zeros((len(block_ids), len(block_ids)))
    for k, b in block_objs.items():
        for dep in b['args']['deps']:
            adj_M[block_ids[dep]][block_ids[k]] = 1

    # get initial nodes without dependencies
    frontier = deque(np.where(adj_M.sum(axis=0) == 0)[0].tolist())
    sorted_dag = []

    while frontier:
        elt = frontier.pop()
        sorted_dag.append(elt)
        dependants = np.where(adj_M[elt, :])[0]
        for dep in dependants:
            adj_M[elt, dep] = 0
            if not adj_M.sum(axis=0)[dep]:
                frontier.append(dep)

    if adj_M.sum() != 0:
        raise Exception("Cycle detected in analysis blocks dependencies, please review and try again!")

    sorted_blocks = []
    for idx in sorted_dag:
        sorted_blocks.append(block_objs[idx2block[idx]])

    return sorted_blocks
