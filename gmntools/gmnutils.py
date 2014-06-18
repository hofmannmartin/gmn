"""
Python package for quantifying and bounding genuine multiparticle entanglement
based on the genuine multiparticle negativity.

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

import numpy
from numpy import array, r_, reshape, transpose, arange, prod, linalg, eye, zeros
import itertools
from itertools import product

class hoperator(object):
	"""
	Create a Hermitian opertor.
	
	matrix : array_like
		An object whose __array__ method returns a two dimensional square array 
	subsystems : list
		A list constaining the dimensions of the subsystems.
		The product of these dimensions have to coinside with the
		number of rows and collumns of the matrix.

	>>> x = hoperator([[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]],[2,2])
	"""
	def __init__(self, matrix, subsystems):
		self.matrix = array(matrix, dtype=numpy.complex128)
		self.__dim = self.matrix.shape
		self._dimH = self.matrix.shape[0]
		self.__dimsubs = tuple(subsystems)
		self.__nsys = len(subsystems)
		if self.__dim[0] != self.__dim[1]:
			raise TypeError("Matrix has to be a square matrix.")
		if [i for i in (self.matrix-self.matrix.conjugate().transpose()).flat if abs(i) > 1e-12]:
			raise TypeError("Matrix has to be Hermitian.")
		if self._dimH != prod(self.__dimsubs):
			raise TypeError("Matrix dimension is %d but expected to be %d by multiplying the dimensions of the subsystems." %(self._dimH,prod(self.__dimsubs)))
	def ptranspose(self,subsys):
		"""
		Partial transpose.
		Returns the partial transpose of the state.

		subsys : list
			A list constaining the subsytems to transpose partially.

		The original partial transposition function 
		was provided by Ville Bergholm (see http://qit.sourceforge.net/) 
		and licenced under the terms of the GPL3.

		>>> x = hoperator([[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]],[2,2])
		
		Transpose subsytem A
		>>> x.ptranspose([0])
		array([[ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
		       [ 0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j],
		       [ 0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j],
		       [ 0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j]])
		
		Transpose subsystem B
		>>> x.ptranspose([1])
		array([[ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
		       [ 0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j],
		       [ 0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j],
		       [ 0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j]])

		Transpose all subsytsems (usual transposition)
		>>> x.ptranspose([0,1])
		array([[ 1.+0.j,  0.+0.j,  0.+0.j,  1.+0.j],
		       [ 0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
		       [ 0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
		       [ 1.+0.j,  0.+0.j,  0.+0.j,  1.+0.j]])
		"""
		# which systems to transpose
		subsys = array(list(set(range(self.__nsys)).intersection(set(subsys))), int)
		# swap the transposed dimensions
		perm = arange(2 * self.__nsys)  # identity permutation
		perm[r_[subsys, subsys + self.__nsys]] = perm[r_[subsys + self.__nsys, subsys]]
		# flat matrix into tensor, partial transpose, back into a flat matrix
		res = self.matrix.reshape(self.__dimsubs + self.__dimsubs).transpose(perm).reshape(self.__dim)
		return res

class densitymatrix(hoperator):
	"""
	Create a densitymatrix.
	
	matrix : array_like
		An object whose __array__ method returns a two dimensional square array
		with unit trace.
	subsystems : list
		A list constaining the dimensions of the subsystems.
		The product of these dimensions have to coinside with the
		number of rows and collumns of the matrix.

	>>> x = hoperator([[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]],[2,2])
	"""
	def __init__(self, matrix, subsystems):
		hoperator.__init__(self,matrix,subsystems)
		if (numpy.abs(numpy.trace(self.matrix)) -1) > 1e-12:
			raise TypeError("Trace of matrix is expected to be 1.")
	def ppt(self,subsys):
		"""
		Test for positive partial transpose [1].
		[1] A. Peres, Phys. Rev. Lett. 77, 1413 (1996).

		subsys : list
			A list constaining the subsytems of one part of the splitting.
		
		>>> x = densitymatrix([[1./2,0,0,1./2],[0,0,0,0],[0,0,0,0],[1./2,0,0,1./2]],[2,2])
		
		Positivity with respect to the splitting A|B
		>>> x.ppt([0])
		False

		>>> x = densitymatrix([[1./2,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1./2]],[2,2])
		
		Positivity with respect to the splitting A|B
		>>> x.ppt([0])
		True
		"""
		try:
			numpy.linalg.cholesky(self.ptranspose(subsys)+1e-100*numpy.eye(self._dimH))
		except numpy.linalg.LinAlgError:
			return False
		return True
	def negativity(self,subsys):
		"""
		Negativity
		Returns the bipartite negativity [1,2] with respect to
		the splitting (subsytsems|rest of the system).
 		[1] K. Zyczkowski, et al., Phys. Rev. A 58, 883 (1998);
		[2] G. Vidal and R.F. Werner, Phys. Rev. A 65, 1 (2002).
		
		subsys : list
			A list constaining the subsytems of one part of the splitting.		
		
		>>> x = densitymatrix([[1./2,0,0,1./2],[0,0,0,0],[0,0,0,0],[1./2,0,0,1./2]],[2,2])

		The negativity with respect to the splitting A|B
		>>> x.negativity([0])
		0.5
		"""
		ev = linalg.eigvalsh(self.ptranspose(subsys))
		return numpy.sum(numpy.absolute(ev[ev < 0]))

class randomlocalmeas(hoperator):
	"""
	Create random local measurements.
	
	subsystems : list
		A list constaining the dimensions of the subsystems.

	The measurement operators are randomly created with respect to local hermitian operator bases

	Initialize class
	>>> A = randomlocalmeas([2,2])

	create random local Measurement
	>>> A.random()
	array([[-0.50414805+0.j        ,  0.85156812-0.64451803j,
	        -0.10785329-0.10199746j,  0.31257410+0.03440337j],
	       [ 0.85156812+0.64451803j, -1.14247832+0.j        ,
	         0.05178087+0.31016915j, -0.24441243-0.23114219j],
	       [-0.10785329+0.10199746j,  0.05178087-0.31016915j,
	         0.13704859-0.j        , -0.23149194+0.17520704j],
	       [ 0.31257410-0.03440337j, -0.24441243+0.23114219j,
	        -0.23149194-0.17520704j,  0.31057354-0.j        ]])
	"""
	def __init__(self, subsystems):
		self.__dimsubs = tuple(subsystems)
		self.__initlocalbases()
	def __initlocalbases(self):
		self.bases = []
		for i in self.__dimsubs:
			localbasis = []
			for j in range(i):
				tmp = zeros((i,i), dtype=numpy.complex128)
				tmp[j,j] = 1.
				localbasis += [tmp]
			for x in arange(i):
				for y in arange(x+1,i):
					tmp = zeros((i,i), dtype=numpy.complex128)
					tmp[x,y] = tmp[y,x] = 1.
					localbasis += [tmp]
					tmp = zeros((i,i), dtype=numpy.complex128)
					tmp[x,y] = -1.j
					tmp[y,x] = 1.j
					localbasis += [tmp]
			self.bases += [localbasis]
	def __tensor(self,*operators):
		if len(operators)==1:
			return operators[0]
		tensor_product = operators[0]
		for i in range(1,len(operators)):
			tensor_product = numpy.kron(tensor_product,operators[i])
		return tensor_product
	def random(self):
		localmeas = []
		for basis in self.bases:
			tmp = numpy.sum([numpy.random.randn()*element for element in basis], axis=0)
			localmeas += [tmp]
		hoperator.__init__(self,self.__tensor(*localmeas),self.__dimsubs)
		return self.matrix


class randomstates(densitymatrix):
	"""
	Create a random densitymatrix.
	
	subsystems : list
		A list constaining the dimensions of the subsystems.

	The matrices randomly created are equaly distributed with respect to the Hilber-Schmidt norm.

	Initialize class
	>>> x = randomstates([2,2])

	create random matrix
	>>> x.random()
	array([[ 0.07913529 +8.33011604e-19j, -0.09735346 +5.05801685e-02j,
 	        0.02347177 +1.03807718e-01j,  0.06246448 -4.72954205e-02j],
	       [-0.09735346 -5.05801685e-02j,  0.34106952 +8.57859005e-19j,
	         0.07168098 -1.55657773e-01j, -0.07911164 -8.13347089e-02j],
	       [ 0.02347177 -1.03807718e-01j,  0.07168098 +1.55657773e-01j,
	         0.40291210 +7.97647640e-19j, -0.08698368 -1.29914600e-01j],
	       [ 0.06246448 +4.72954205e-02j, -0.07911164 +8.13347089e-02j,
	        -0.08698368 +1.29914600e-01j,  0.17688309 -2.48851825e-18j]])
	"""
	def __init__(self, subsystems):
		self.__dimsubs = tuple(subsystems)
		self.random()
	def random(self):
		n = prod(self.__dimsubs)
		M = numpy.random.randn(n,n)+1.j*numpy.random.randn(n,n)
		matrix = numpy.dot(M.conjugate().transpose(),M)
		matrix /= numpy.trace(matrix)
		densitymatrix.__init__(self,matrix,self.__dimsubs)
		return self.matrix

class pauli():
	"""
	Create a n qubit Pauli operator basis.
	
	nsys : int
		The total number of qubits the system is compose of.

	This class allows to provide a basis of tensor products of Pauli operators and identity. Furthermore, it can automatically detect generalized stabilizer symmetries of a state and provide a symmetric operatorbasis.

	Initialize class
	>>> x = pauli(2)

	Iterate over the two qubit Pauli basis
	>>> print [i for i in pauli(1)]
		[array([[ 1.+0.j,  0.+0.j],
			[ 0.+0.j,  1.+0.j]]), array([[ 0.+0.j,  1.+0.j],
			[ 1.+0.j,  0.+0.j]]), array([[ 0.+0.j,  0.-1.j],
			[ 0.+1.j,  0.+0.j]]), array([[ 1.+0.j,  0.+0.j],
			[ 0.+0.j, -1.+0.j]])]
	"""
	def __init__(self,nsys):
		self.n = nsys
		self.e = array(eye(2),dtype=numpy.complex128)
		self.x = array([[0,1],[1,0]],dtype=numpy.complex128)
		self.y = array([[0,-1.j],[1.j,0]],dtype=numpy.complex128)
		self.z = array([[1,0],[0,-1]],dtype=numpy.complex128)
	def __tensor(self,*operators):
		if len(operators)==1:
			return operators[0]
		tensor_product = operators[0]
		for i in range(1,len(operators)):
			tensor_product = numpy.kron(tensor_product,operators[i])
		return tensor_product
	def __iter__(self):
		for i in product(['e','x','y','z'],repeat=self.n):
			yield self.operator(''.join(i))
	def __itersymofrho(self,rho):
		coef = [numpy.trace(numpy.dot(i,array(rho,dtype=numpy.complex128))).real for i in self.basis()]
		maxval = numpy.max(coef[1:])
		coef = [True  if numpy.abs(i) > maxval*1e-10 else False for i in coef]
		coef = [i for i in itertools.compress([''.join(i) for i in product(['e','x','y','z'],repeat=self.n)],coef)]
		return [''.join(i) for i in product(['e','x','y','z'],repeat=self.n) if self.__commute(''.join(i),coef)]
	def __commute(self, op, sym):
		for i in sym:
			c=1
			for char1,char2 in zip(op,i):
				if char1 != char2 and char1 != 'e' and char2 != 'e':
					c *= -1
			if c == -1:
				return False
		return True
	def basis(self):
		"""
		Returns the n qubits Pauli operator basis.
		
		Iterate over the two qubit Pauli basis
		>>> pauli(1).basis()
			[array([[ 1.+0.j,  0.+0.j],
				[ 0.+0.j,  1.+0.j]]), array([[ 0.+0.j,  1.+0.j],
				[ 1.+0.j,  0.+0.j]]), array([[ 0.+0.j,  0.-1.j],
				[ 0.+1.j,  0.+0.j]]), array([[ 1.+0.j,  0.+0.j],
				[ 0.+0.j, -1.+0.j]])]
		"""
		return [i for i in self.__iter__()]
	def symbasis(self,rho):
		"""
		Returns a symmetric operator basis if possible
		
		Initialize Bell state
		>>> rho = numpy.zeros((4,4),dtype=numpy.complex128)
		>>> rho[0,0] = rho[0,-1] = rho[-1,0] = rho[-1,-1] = .5

		Initialize two qubit pauli class
		>>> x = pauli(2)

		Calculate symmetric basis ['ee','xx','yy','zz']
		>>> x.symbasis(rho)
		[array([[ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
		       [ 0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j],
		       [ 0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j],
		       [ 0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j]]), array([[ 0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j],
		       [ 0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j],
		       [ 0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j],
		       [ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j]]), array([[ 0.+0.j,  0.+0.j,  0.+0.j, -1.-0.j],
		       [ 0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j],
		       [ 0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j],
		       [-1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j]]), array([[ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
		       [ 0.+0.j, -1.+0.j,  0.+0.j, -0.+0.j],
		       [ 0.+0.j,  0.+0.j, -1.+0.j, -0.+0.j],
		       [ 0.+0.j, -0.+0.j, -0.+0.j,  1.-0.j]])]
		"""
		sym = self.__itersymofrho(rho)
		return [self.operator(''.join(i)) for i in product(['e','x','y','z'],repeat=self.n) if self.__commute(''.join(i),sym)]
	def operator(self,seq):
		"""
		Returns the n fold tensor product of Pauli operator as array.
		
		seq : string
			a string representing the input operators to tensor.
			for identity use either of 'e', 'E' or '0'
			for sigma_x use either of 'x', 'X' or '1'
			for sigma_y use either of 'y', 'Y' or '2'
			for sigma_z use either of 'z', 'Z' or '3'
	
		Return ZYX
		>>> pauli(3).operator('zyx')
			array([[ 0.+0.j  0.+0.j  0.+0.j  0.-1.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
			 [ 0.+0.j  0.+0.j  0.-1.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
			 [ 0.+0.j  0.+1.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
			 [ 0.+1.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
			 [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j -0.+0.j -0.+0.j  0.+0.j  0.+1.j]
			 [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j -0.+0.j -0.+0.j  0.+1.j  0.+0.j]
			 [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.-0.j  0.-1.j -0.+0.j -0.+0.j]
			 [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.-1.j  0.-0.j -0.+0.j -0.+0.j]])
		"""
		seq = str(seq)
		sequence = []
		for i in seq:
			if i in ['e','E','0']:
				sequence += [self.e]
			if i in ['z','Z','1']:
				sequence += [self.z]
			if i in ['y','Y','2']:
				sequence += [self.y]
			if i in ['x','X','3']:
				sequence += [self.x]
		return self.__tensor(*sequence)

# End of operator.py
