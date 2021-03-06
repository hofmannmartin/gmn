from distutils.core import setup

setup(name='gmntools',
	version='1.0.0',
	description='A numerical implementation of the genuine multiparticle negativity',
	long_description=open('README.md').read(),
	author='Martin Hofmann',
	author_email='hofmann@physik.uni-siegen.de',
	url='https://github.com/hofmannmartin/gmn',
	packages=['gmntools'],
	license = 'GNU GPL version 3',
	requires = ['cvxopt']
     )
