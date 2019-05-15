from setuptools import setup

setup(name='xspline',
      version='0.0.0',
      description='robust spline package',
      url='https://github.com/zhengp0/bspline',
      author='Peng Zheng',
      author_email='zhengp@uw.edu',
      license='MIT',
      packages=['xspline'],
      install_requires=['numpy', 'scipy'],
      zip_safe=False)
