"""
For .py in main, use 'import xhd_source as xhds'.
If use 'from xhd_source import *', the whole folder will be imported.
Only .py files are supported in this folder.
"""

import os

# Set __all__ to be all .py files in the folder
dir_path = os.path.dirname(os.path.realpath(__file__))
filenames = os.listdir(dir_path)
splitted_file_names = [i.split('.') for i in filenames]
py_modules = [i[0] for i in splitted_file_names if len(i) == 2 and i[-1] == 'py' and i[0] != '__init__']
__all__ = py_modules
