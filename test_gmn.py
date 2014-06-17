#!/usr/bin/python
# Filename: Operator.py

import numpy
from numpy import random, array
import itertools
from itertools import chain
import gmnutils
from gmnutils import hoperator, densitymatrix, pauli
import gmntools
from gmntools import gmn
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
			self.assertListEqual([i for i in chain(*self.hmatrix)], [i for i in chain(*self.test.matrix)])
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
		self.assertListEqual([i for i in chain(*self.pttest.ptranspose([1]))],[i for i in chain(*self.pttest.ptranspose([0]))])
		self.assertListEqual([1,2,3,6,2,5,4,7,3,4,8,9,6,7,9,10],[i for i in chain(*self.pttest.ptranspose([0]))])
		self.assertListEqual([1,2,3,4,2,5,6,7,3,6,8,9,4,7,9,10],[i for i in chain(*self.pttest.ptranspose([0,1]))])

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
		self.assertAlmostEqual(0.5,densitymatrix(self.ghz,self.subsystems).negativity([0]))
		self.assertEqual(self.neg,self.rho.negativity([0]))

class Test_Pauli(unittest.TestCase):
	def setUp(self):
		self.n = random.randint(1,5)
		self.pauli = pauli(self.n)
		rho = numpy.zeros((2**self.n,2**self.n),dtype=numpy.complex128)
		rho[0,0] = rho[0,-1] = rho[-1,0] = rho[-1,-1] = .5
		self.rho = rho
		self.zy = self.pauli.operator('zy')
	
	def test_pauli(self):
		# make sure class is correctly initialized
		self.assertEqual(self.n,self.pauli.n)
		self.assertEqual(len([0 for i in self.pauli]),4**self.n)
		self.assertEqual(len(self.pauli.basis()),4**self.n)
		# Test pauli.operator function
		self.assertListEqual([i for i in chain(*self.pauli.operator('zy'))],[i for i in chain(*self.zy)])
		# Test if symmetric basis is obtained correctly
		self.assertEqual(len(self.pauli.symbasis(self.rho)),2**self.n)

class Test_GMN(unittest.TestCase):
	def setUp(self):
		self.subsys = [2,2,2]
		self.gmn = gmn([2,2,2])
		rho = numpy.zeros((8,8),dtype=numpy.complex128)
		rho[0,0] = rho[0,-1] = rho[-1,0] = rho[-1,-1] = .5
		self.ghz = rho
		psi = numpy.zeros((8,),dtype=numpy.complex128)
		psi[1] = psi[2] = psi[4] = 1.
		self.w = numpy.outer(psi,psi)/3.
		self.pauli = pauli(3)
		measurements = [self.pauli.operator(i) for i in ['xxz','xzx','zxx','zzz']]
		self.measurements = [(o,numpy.trace(numpy.dot(self.w,o).real)) for o in measurements]

	def test_partinfo(self):
		# test if W state gets detected using the partial information of the 4 measurements XXZ, XZX, ZXX, XXZ by trying to construct a fully decomposable witness
		self.assertAlmostEqual(0.25, self.gmn.gmn_partial_info(self.measurements),places=6)
		# test if W state does not get detected using the partial information of the 4 measurements XXZ, XZX, ZXX, ZZZ by trying to construct a fully PPT witness
		self.assertAlmostEqual(0, self.gmn.gmn_partial_info_ppt(self.measurements))
	
	def test_gmninit(self):
		# make sure class is correctly initialized
		self.assertEqual(self.gmn.rhom,{})
		self.assertEqual(self.gmn.W,{})
		self.assertEqual(self.gmn.status,'')
		# should raise an exception if no matrix is provided
		with self.assertRaises(Exception):
			self.gmn.gmn()

	def test_gmn(self):
		# Test if genuine multiparticle negativity of GHZ state is 0.5
		self.gmn.setdensitymatrix(self.ghz)
		self.assertAlmostEqual(self.gmn.gmn(), 0.5)
		self.assertEqual(self.gmn.status, 'optimal')
		self.assertAlmostEqual(self.gmn.gmn_jmg(), 0.5)
		self.assertEqual(self.gmn.status, 'optimal')
		# Test if genuine multiparticle negativity of GHZ state is 0.5 with symmetric operatorbasis
		self.gmn.setsympaulibasis()
		self.assertAlmostEqual(self.gmn.gmn(), 0.5)
		self.assertEqual(self.gmn.status, 'optimal')
		self.assertAlmostEqual(self.gmn.gmn_jmg(), 0.5)
		# Test if genuine multiparticle negativity of GHZ state is 0.5 with symmetric operatorbasis and real Witnesses
		self.assertAlmostEqual(self.gmn.gmn(real=True), 0.5, places=6)
		self.assertEqual(self.gmn.status, 'optimal')
		self.assertAlmostEqual(self.gmn.gmn_jmg(real=True), 0.5, places=6)
		# Test if genuine multiparticle negativity of GHZ state is 0.5 with symmetric operatorbasis, real Witnesses and alternative solver
		self.assertAlmostEqual(self.gmn.gmn(real=True,altsolver='dsdp'), 0.5, places=6)
		self.assertEqual(self.gmn.status, 'optimal')
		self.assertAlmostEqual(self.gmn.gmn_jmg(real=True,altsolver='dsdp'), 0.5, places=6)
		# Test if genuine multiparticle negativity of GHZ state is 0.5 with real operatorbasisand real Witnesses
		self.gmn.setrealbasis()
		self.assertAlmostEqual(self.gmn.gmn(real=True), 0.5, places=6)
		self.assertEqual(self.gmn.status, 'optimal')
		self.assertAlmostEqual(self.gmn.gmn_jmg(real=True), 0.5, places=6)
		self.assertEqual(self.gmn.status, 'optimal')
		# Test if optimal witness gives the right expectation value
		self.assertAlmostEqual(self.gmn.witness_expectationvalue(), 0.5, places=6)
		# Test if genuine multiparticle negativity of W state is sqrt(2)/3 with real operatorbasisand real Witnesses
		self.gmn.setdensitymatrix(self.w)
		self.gmn.setsympaulibasis()
		self.assertAlmostEqual(self.gmn.gmn(real=True,altsolver='dsdp'), numpy.sqrt(2)/3.)
		self.assertEqual(self.gmn.status, 'optimal')
		self.assertAlmostEqual(self.gmn.gmn_jmg(real=True,altsolver='dsdp'), 0.4428090)
		self.assertEqual(self.gmn.status, 'optimal')

	def test_gmn_oldnew(self):
		self.gmn.setdensitymatrix(self.w)
		self.gmn.setsympaulibasis()
		# test if the genuine multiparticle negativity (gmn_jmg) is less equal then renormalized version (gmn) @ W-state
		self.assertLessEqual(self.gmn.gmn_jmg(real=True,altsolver='dsdp'), self.gmn.gmn(real=True,altsolver='dsdp'))
		# test if the genuine multiparticle negativity (gmn_jmg) is less equal then renormalized version (gmn) @ GHZ-state
		self.gmn.setdensitymatrix(self.ghz)
		self.gmn.setsympaulibasis()
		self.assertAlmostEqual(self.gmn.gmn_jmg(real=True,altsolver='dsdp'), self.gmn.gmn(real=True,altsolver='dsdp'))

if __name__ == '__main__':
    unittest.main()

# End of operator.py
