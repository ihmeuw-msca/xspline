from setuptools import setup

setup(name='xspline',
      version='0.0.4',
      description='robust spline package',
      url='https://github.com/zhengp0/xspline',
      author='Peng Zheng',
      author_email='zhengp@uw.edu',
      license='MIT',
      packages=['xspline'],
      package_dir={'xspline': 'src/xspline'},
      install_requires=['numpy', 'scipy', 'pytest'],
      zip_safe=False)
