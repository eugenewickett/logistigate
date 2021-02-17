from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
import pathlib

here = pathlib.Path(__file__).parent.resolve()
# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

class Run_TestSuite(TestCommand):
    def run_tests(self):
        #import os
        import sys
        py_version = sys.version_info[0]
        print('Python version from setup.py is', py_version)
        #run_string = "tests/run-tests.sh -p " + str(py_version)
        #os.system(run_string)

setup(
    name='logistigate',
    version='0.1.0',  # Required
    author='Eugene Wickett, Karen Smilowitz, Matthew Plumlee',
    author_email='eugenewickett@u.northwestern.edu',
    description='python implementation of logistigate for supply-chain aberration inference',
    long_description=long_description,
    long_description_content_type='text/plain',
    url='https://github.com/eugenewickett/logistigate',
    #package_dir={'logistigate': 'logistigate'},
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha', #   3 - Alpha, 4 - Beta, 5 - Production/Stable
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',        
    ],
    keywords='supply chains, statistical inference, pharmaceutical regulation',
    python_requires='>=3.4',
    install_requires=['nuts','numpy','tabulate',
                      'scipy','matplotlib'],
    #packages=['mypkg'],
    package_dir={'logistigate': 'src/logistigate'},
    package_data={'logistigate': ['data/*.csv']},
    include_package_data=True
    #cmdclass={'test': Run_TestSuite}
)
