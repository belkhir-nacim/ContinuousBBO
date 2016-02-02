from distutils.core import setup

setup(
    name='benchmark',
    version='1',
    packages=['benchmark', 'benchmark.BBOB', 'benchmark.CEC2005_real', 'benchmark.CEC2013_large_scale'],
    url='',
    license='',
    author='nacimbelkhir',
    author_email='nacim.belkhir@thalesgroup.com,nacim.belkhir@inria.fr',
    description='A set of Black Box test benchs for continuous optimization in mono and multi objective',
    requires=['numpy','diversipy']
)
