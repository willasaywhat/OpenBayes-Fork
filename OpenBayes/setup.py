from distutils.core import setup
import sys

# patch distutils if it can't cope with the "classifiers" or
# "download_url" keywords
if sys.version < '2.2.3':
    from distutils.dist import DistributionMetadata
    DistributionMetadata.classifiers = None
    DistributionMetadata.download_url = None

setup(  name='OpenBayes',
        version='0.1.0',
        description='An OpenSource Python implementation of bayesian networks inspired by BNT.',
        author = 'Kosta Gaitanis, Elliot Cohen',
	    author_email = 'gaitanis@tele.ucl.ac.be, elliot.cohen@gmail.com',
	    url = 'http://www.openbayes.org',
        packages = ['OpenBayes'],
        package_dir = {'OpenBayes':'.', 'OpenBayes.Examples':'./Examples'},
	    license = 'modified Python',
	    classifiers=[
          'Development Status :: 4 - Beta',
          'Environment :: Console',
          'Intended Audience :: Science/Research',
          'Intended Audience :: Developers',
          'License :: Free for non-commercial use',
	  'Operating System :: OS Independent',
          'Programming Language :: Python',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Topic :: Software Development :: Libraries :: Python Modules',
          ],
	long_description = """OpenBayes is a library that allows users to easily create a bayesian network and perform inference on it.
It is mainly inspired from the Bayes Net Toolbox (BNT) which is available for MatLAB, 
but uses python as a base language which provides many benefits : fast execution, portability 
and ease to use and maintain. Any inference engine can be implemented by inheriting a base
class. In the same way, new distributions can be added to the package by simply defining the 
data contained in the distribution and some basic probabilistic operations. 

The project is mature enough to be used for static bayesian networks and we are currently 
developping the dynamical aspect."""
)
