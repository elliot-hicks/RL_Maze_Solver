from setuptools import setup, find_packages

setup(name='maze_maker',
      version='0.0.1',
      packages = find_packages(include = ['maze_maker_package', 'maze_maker_package.*'])
)