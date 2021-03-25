from setuptools import setup, find_packages

VERSION = '0.0.2' 
DESCRIPTION = 'Imp. of EM Clustering Algorithm'
LONG_DESCRIPTION = 'Implementation of the EM Clustering Algorithm for arbitrary models'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="timemclust", 
        version=VERSION,
        author="Mason Watson",
        author_email="msn.watson@gmail.com",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[
            'ticks',
            'scipy',
            'numpy',
            'copy',
            'abc',
            'collections',
            'matplotlib'
        ],
        
        keywords=['python', 'clustering', 'time series'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
        ]
)