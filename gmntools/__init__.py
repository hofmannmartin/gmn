"""
Python package for quantifying and bounding genuine multiparticle entanglement

GMNTOOLS is a free software package quantifying and bounding genuine 
multiparticle entanglement based on the Python programming language.
It can be used with the interactive Python interpreter, on the 
command line by executing Python scripts, or integrated in other 
software via Python extension modules. Its main purpose is to make 
the quantification of genuine multiparticle entanglement easy for
Python beginners and provide a good basis for experts.
"""

# Copyright 2014 Martin Hofmann.
#
# This file is part of GMNTOOLS version 1.0.
#
# GMNTOOLS is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# GMNTOOLS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License

from gmntools import gmn
from gmnutils import hoperator, densitymatrix, randomlocalmeas, randomstates, pauli

__all__ = [ 'gmn', 'hoperator', 'densitymatrix', 'randomlocalmeas', 'randomstates', 'pauli' ]
