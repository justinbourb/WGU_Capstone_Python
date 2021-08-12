import pathlib
import os
current_dir = os.getcwd()
print(current_dir.split('\\')[-1] == 'dash app')
os.chdir('../../data')
dir = os.getcwd()
print(dir.split('\\')[-1])