import os
import unittest


class TestReamde(unittest.TestCase):
    # Interesting: has coordinates as inputs
    def test_0_all_readme_code(self):
        dirname = os.path.dirname(__file__)
        readme_fn = os.path.join(dirname, '..', '..', 'README.md')
        with open(readme_fn, 'r') as fp:
            readme = fp.read()
        code = ''
        for i, block in enumerate(readme.split('```')):
            if i % 2 == 1:
                code += block + '\n'
        code = code.replace('python', '').replace('pip3 install lightwood', '').replace('pip install lightwood', '')
        
        exec(code)
