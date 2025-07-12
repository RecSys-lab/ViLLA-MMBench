from setuptools import setup, find_packages

# Setup configuration for the ViLLA-MMBench package
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt', encoding='utf-8') as f:
    requirements = f.read().splitlines()

setup(
    name='villa_mmbench',
    version='1.0.0',
    author='Ali Tourani',
    author_email='ali.tourani@uni.lu',
    description='A framework for benchmarking multimodal models for video recommendation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/RecSys-lab/ViLLA-MMBench',
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
