from distutils.core import setup

setup(name='boring_battery',
      version='1.0.0',
      packages=[
          'src'
      ],
      py_modules=['example1'],
      install_requires=[
        'openmdao>=2.0.0',
      ]
)
