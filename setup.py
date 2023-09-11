import setuptools
from setuptools import setup

from ezmodel.version import __version__

__name__ = "ezmodel"
__author__ = "Julian Blank"
__url__ = ""

data = dict(
    name=__name__,
    version=__version__,
    author=__author__,
    url=__url__,
    python_requires='>=3.6',
    author_email="blankjul@msu.edu",
    description="Machine Learning, Model, Surrogate, Metamodels, Response Surface",
    license='Apache License 2.0',
    keywords="model",
    install_requires=['numpy', "pandas", "scipy", "scikit-learn", "pydacefit", "matplotlib"],
    include_package_data=True,
    platforms='any',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics'
    ]
)


# ---------------------------------------------------------------------------------------------------------
# OTHER METADATA
# ---------------------------------------------------------------------------------------------------------


# update the readme.rst to be part of setup
def readme():
    with open('README.rst') as f:
        return f.read()


def packages():
    return ["ezmodel"] + ["ezmodel." + e for e in setuptools.find_packages(where='ezmodel')]


data['long_description'] = readme()
data['long_description_content_type'] = 'text/x-rst'
data['packages'] = packages()

# ---------------------------------------------------------------------------------------------------------
# SETUP
# ---------------------------------------------------------------------------------------------------------

setup(**data)
