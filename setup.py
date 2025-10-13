from setuptools import setup, find_packages

setup(name='S5',
      version='0.1',
      description='Simplified State Space Models for Sequence Modeling.',
      author='J.T.H. Smith, A. Warrington, S. Linderman.',
      author_email='jsmith14@stanford.edu',
      packages=find_packages(include=['s5', 's5.*', 'lob', 'lob.*']),
     )
