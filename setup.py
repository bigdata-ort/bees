from setuptools import setup, find_packages

setup(
    name='bees',
    version='0.1',
    packages=find_packages(exclude=[]),
    license='MIT',
    description='Bees',
    long_description=open('README.txt').read(),
    install_requires=['numpy'],
    url='https://github.com/bigdata-ort/bees',
    author='Bigdata ORT',
    author_email='yovine@ort.edu.uy'
)
