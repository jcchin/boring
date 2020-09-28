from setuptools import setup, find_packages

setup(name='boring_battery',
      version='1.0.0',
      packages=find_packages(),
      py_modules=['example1'],
      install_requires=[
        'openmdao>=2.0.0',
      ]
)
