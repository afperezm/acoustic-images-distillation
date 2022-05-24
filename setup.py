from setuptools import setup

setup(
    name='codebase',
    version='0.0.1',
    packages=['codebase'],
    install_requires=['librosa', 'numpy', 'tensorflow-gpu==2.6.4', 'torchfile'],
    url='https://gitlab.iit.it/aperez/acoustic-images-distillation',
    license='',
    author='Andres Perez',
    author_email='andres.perez@iit.it',
    description='A Python package with a sample data loader, trainers and models'
)
