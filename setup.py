from distutils.core import setup

setup(
    name='EchoMetrics',
    version='0.1.0',
    author='Sam Urmy',
    author_email='urmy@uw.edu',
    packages=['echometrics', 'echometrics.tests'],
    package_data={'echometrics' : ['data/*']},
    #scripts=['bin/stowe-towels.py','bin/wash-towels.py'],
    #url='http://pypi.python.org/pypi/TowelStuff/',
    license='LICENSE.txt',
    description='Acoustic water column metrics',
    long_description=open('README.txt').read(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "pandas"
    ],
)