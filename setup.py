from distutils.core import setup

setup(name='boring_battery',
      version='1.0.0',
      packages=[
          'src',
          'test',
          'util',
          'XDSM'
      ],

      install_requires=[
        'openmdao>=2.0.0',
      ]
)
