#!/usr/bin/python
# Filename: Operator.py

import numpy
from numpy import random, array
import gmnutils
from gmnutils import hoperator, densitymatrix, pauli
import unittest

class Test_hoperator(unittest.TestCase):
	def setUp(self):
		self.subsystems = random.randint(2,5,size=random.randint(4))
		self.dim = numpy.prod(self.subsystems)
		self.sqmatrix = random.randn(self.dim,self.dim) + 1.j*random.randn(self.dim,self.dim)
		self.hmatrix = numpy.dot(self.sqmatrix.conjugate().transpose(),self.sqmatrix)
		self.test = hoperator(self.hmatrix,self.subsystems)
		self.pttest = hoperator([[1,2,3,4],[2,5,6,7],[3,6,8,9],[4,7,9,10]],[2,2])
	
	def test_hoperator(self):
		for i in range(1000):
			# make sure operator is passed to class
			self.assertTrue(not (array(self.hmatrix,dtype=numpy.complex128)-self.test.matrix).all())
			# should raise an exception for non Hermitian matrix
			with self.assertRaises(TypeError):
				hoperator(self.sqmatrix,self.subsystems)

			self.sqmatrix = random.randn(self.dim,self.dim) + 1.j*random.randn(self.dim,self.dim)
			self.hmatrix = numpy.dot(self.sqmatrix.conjugate().transpose(),self.sqmatrix)
			self.test = hoperator(self.hmatrix,self.subsystems)
		
		# should raise an exception for non square matrix
		with self.assertRaises(TypeError):
			hoperator(self.hmatrix[:-1,:],self.subsystems)
		# should raise an exception if dimensions of matrix do not fit product of dimensions of subsystems
		with self.assertRaises(TypeError):
			hoperator(self.hmatrix[:-1,:-1],self.subsystems)
		# make sure operator is properly transposed
		self.assertTrue(not (array([[1,2,3,6],[2,5,4,7],[3,4,8,9],[6,7,9,10]])-self.pttest.ptranspose([0])).all())
		self.assertTrue(not (array([[1,2,3,4],[2,5,6,7],[3,6,8,9],[4,7,9,10]]).transpose()-self.pttest.ptranspose([1,2])).all())

class Test_densitymatrix(unittest.TestCase):
	def setUp(self):
		self.subsystems = [2,2,2]
		M = random.randn(8,8) + 1.j*random.randn(8,8)
		rho = numpy.dot(M.conjugate().transpose(),M)
		rho /= numpy.trace(rho)
		self.rho = densitymatrix(rho,self.subsystems)
		ptrho = numpy.zeros((8,8), dtype=numpy.complex128)
		ptrho[:4,:4] = rho[:4,:4]
		ptrho[4:,4:] = rho[4:,4:]
		ptrho[:4,4:] = rho[4:,:4]
		ptrho[4:,:4] = rho[:4,4:]
		ev =  numpy.linalg.eigvalsh(ptrho)
		self.neg = numpy.sum(numpy.absolute(ev[ev < 0]))
		self.ghz = [[1./2,0,0,0,0,0,0,1./2],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[1./2,0,0,0,0,0,0,1./2]]
		self.sep = [[1./2,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1./2]]
	
	def test_densitymatrix(self):
		# should raise an exception for non unit trace matrices
		with self.assertRaises(TypeError):
			densitymatrix(2*self.rho.matrix,self.subsystems)
		# GHZ state is not PPT
		self.assertEqual(False,densitymatrix(self.ghz,self.subsystems).ppt([0]))
		# Separable state is PPT
		self.assertEqual(True,densitymatrix(self.sep,self.subsystems).ppt([0]))
		# make sure negativity is properly calculated
		self.assertEqual(0.5,densitymatrix(self.ghz,self.subsystems).negativity([0]))
		self.assertEqual(self.neg,self.rho.negativity([0]))

class Test_Pauli(unittest.TestCase):
	def setUp(self):
		self.n = random.randint(1,5)
		self.pauli = pauli(self.n)
		rho = numpy.zeros((2**self.n,2**self.n),dtype=numpy.complex128)
		rho[0,0] = rho[0,-1] = rho[-1,0] = rho[-1,-1] = .5
		self.rho = rho
		self.zy = numpy.kron(array([[1,0],[0,-1]],dtype=numpy.complex128), array([[0,-1.j],[1.j,0]],dtype=numpy.complex128))
	
	def test_pauli(self):
		# make sure class is correctly initialized
		self.assertEqual(self.n,self.pauli.n)
		self.assertEqual(len([0 for i in self.pauli]),4**self.n)
		self.assertEqual(len(self.pauli.basis()),4**self.n)
		self.pauli.operator('ze')
		self.assertTrue(not (self.pauli.operator('ze')-self.zy).all())
		self.assertEqual(len(self.pauli.symbasis(self.rho)),2**self.n)

if __name__ == '__main__':
    unittest.main()

# End of operator.py
