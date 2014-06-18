===
gmn
===

Python package for quantifying and bounding genuine multiparticle entanglement
based on the genuine multiparticle negativity.

GMNTOOLS is a free software package quantifying and bounding genuine 
multiparticle entanglement based on the Python programming language.
It can be used with the interactive Python interpreter, on the 
command line by executing Python scripts, or integrated in other 
software via Python extension modules. Its main purpose is to make 
the quantification of genuine multiparticle entanglement easy for
Python beginners and provide a good basis for experts.

Installation
============

Required and optional software
------------------------------

The package requires version 2.7 or 3.x of Python, and is built from
source, so the header files and libraries for Python must be installed,
as well as the core binaries.

The installation requires the Python convex optimization package CVXOPT
as well as Numpy and Scipy.

The following software libraries are optional.

* DSDP5.8 <www-unix.mcs.anl.gov/DSDP> is a semidefinite programming solver.


Installation procedure
----------------------

GMNTOOLS can be installed globally by writing:

    python setup.py install

It can also be installed locally by typing:

    python setup.py install --user

To test that the installation was successful, go to the test directory
and run the test module provided

     cd examples/test
     python test_gmn.py

If Python does not issue an error message, installation was successful.

Additional information can be found in the
Python documentation <http://docs.python.org/install/index.html>.
