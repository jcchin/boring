from distutils.core import setup

setup(name='boring_battery',
      version='1.0.0',
      packages=[
          'src'
      ],

      install_requires=[
        'openmdao>=2.0.0',
      ]
)
