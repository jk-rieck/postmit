"""Setup postmit."""

from setuptools import setup

setup(name='postmit',
      description='MITgcm output post-processing',
      packages=['postmit'],
      package_dir={'postmit': 'postmit'},
      install_requires=['setuptools', ],
      zip_safe=False)
