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

import gmnutils
from gmnutils import densitymatrix, hoperator, pauli

import cvxopt
from cvxopt.base import spmatrix
from cvxopt import solvers

import numpy
from numpy import array, arange, prod, zeros, eye

# ### set solver options ####
# Set options for standard sdp solver
solvers.options ['show_progress'] = False
#solvers.options ['abstol'] = 1. e -12
#solvers.options ['reltol'] = 1. e -12
#solvers.options ['feastol'] = 1. e -8
# Set options for dsdp solver
solvers.options ['DSDP_Monitor'] = 0 # integer ( default : 0)
#solvers.options ['DSDP_GapTolerance'] = 1e -12 # scalar ( default : 1e -5) .


class gmn():
	"""
	The genuine multiparticle negativity toolbox is a collection of methods 
	to detect and quantify genuine multiparticle entanglement in mixed states
	using fully decomposable witnesses [1]. It incorparates most of the functionality
	of the MATLAB program PPTMixer. It extends its functionality to cases, where
	only partial information on a state is known and contains also the renormalized
	version of the genuine multiparticle negativity [2].
	[1] B. Jungnitsch, T. Moroder, and O. Guhne, Phys. Rev. Lett. 106, 190502 (2011).
	[2] M. Hofmann, T. Moroder, and O. Guhne, J. Phys. A: Math. Theor. 47 155301 (2014).
	
	subsystems : list
		A list constaining the dimensions of the subsystems. The product of these
		dimensions have to coinside with the number of rows and	collumns of the matrix.
	matrix : array_like
		An array_like object representing a density matrix of the given system.
	opbasis : list of array_like
		A list of Hermitian matrices spanning the set of all bound operators on the
		Hilbertspace or some subspace thereof.

	To calculate the genuine multiparticle negativity of the GHZ state:
	1.) initialize the class.
	>>> from gmntools import gmn
	>>> gmntool = gmn([2,2,2],[[.5,0,0,0,0,0,0,.5],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[.5,0,0,0,0,0,0,.5]])
	
	2.) call the memberfunction gmn_jmg (original version [1]) or gmn (renormalized [2]).
	>>> gmntool.gmn()
	0.499999998369068
	"""
	def __init__(self,subsystems,matrix=None,opbasis=[]):
		self.__dim = (prod(subsystems),prod(subsystems))
		self.__dimH = prod(subsystems)
		self.__dimsubs = tuple(subsystems)
		self.__nsys = len(subsystems)
		self.W_part_info = zeros(self.__dim, dtype=numpy.complex128)
		self.W = {}
		self.rhom = {}
		self.status = ''
		if matrix != None:
			self.setdensitymatrix(matrix)
		else:
			self.__rho = None
		self.setoperatorbasis(opbasis)
	def __initoperatorbasis(self):
		self.operatorbasis = []
		for i in arange(self.__dimH):
			tmp = zeros(self.__dim, dtype=numpy.complex128)
			tmp[i,i] = 1.
			self.operatorbasis += [hoperator(tmp,self.__dimsubs)]
		for x in arange(self.__dimH):
			for y in arange(x+1,self.__dimH):
				tmp = zeros(self.__dim, dtype=numpy.complex128)
				tmp[x,y] = tmp[y,x] = 1.
				self.operatorbasis += [hoperator(tmp,self.__dimsubs)]
				tmp = zeros(self.__dim, dtype=numpy.complex128)
				tmp[x,y] = -1.j
				tmp[y,x] = 1.j
				self.operatorbasis += [hoperator(tmp,self.__dimsubs)]
	def __initrealoperatorbasis(self):
		self.operatorbasis = []
		for i in arange(self.__dimH):
			tmp = zeros(self.__dim, dtype=numpy.complex128)
			tmp[i,i] = 1.
			self.operatorbasis += [hoperator(tmp,self.__dimsubs)]
		for x in arange(self.__dimH):
			for y in arange(x+1,self.__dimH):
				tmp = zeros(self.__dim, dtype=numpy.complex128)
				tmp[x,y] = tmp[y,x] = 1.
				self.operatorbasis += [hoperator(tmp,self.__dimsubs)]
	def witness_expectationvalue(self,rho=None):
		"""
		Witness Expectationvalue
		Returns the expectation value of a prior compute witness.

		rho : array_like
			An array_like object representing a density matrix of the given system.
		
		As an example we use the GHZ witness and try to detect the W state:
		>>> from gmntools import gmn
		
		Set GHZ and W state
		>>> ghz = [[.5,0,0,0,0,0,0,.5],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[.5,0,0,0,0,0,0,.5]]
		>>> w = [[0,0,0,0,0,0,0,0],[0,1./3.,1./3.,0,1./3.,0,0,0],[0,1./3.,1./3.,0,1./3.,0,0,0],[0,0,0,0,0,0,0,0],[0,1./3.,1./3.,0,1./3.,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]]
		
		Call the gmn memberfunction to calculate the optimal witness for the GHZ state
		>>> gmntool = gmn([2,2,2],ghz)
		>>> gmntool.gmn()
		0.499999998369068

		The optimal witness for the GHZ state does not detect the W state
		>>> gmntool.witness_expectationvalue(w)
		-0.88501744664092974
		"""
		if not self.W:
			raise TypeError("Memberfuction availible after using Memberfunctin gmn only.")
		if rho==None:
			return -numpy.trace(numpy.dot(self.W['W'],self.__rho.matrix)).real
		else:
			return -numpy.trace(numpy.dot(self.W['W'],rho)).real
	def setdensitymatrix(self,rho):
		"""
		Set the density matrix
		Sets the density matrix for which the genuine mutliparticle negativity should be computed

		rho : array_like
			An array_like object representing a density matrix of the given system.
		
		Evaluate the genuine multiparticlce negativity first for the GHZ state and afterwards for the W state
		>>> from gmntools import gmn
		
		Set GHZ and W state
		>>> ghz = [[.5,0,0,0,0,0,0,.5],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[.5,0,0,0,0,0,0,.5]]
		>>> w = [[0,0,0,0,0,0,0,0],[0,1./3.,1./3.,0,1./3.,0,0,0],[0,1./3.,1./3.,0,1./3.,0,0,0],[0,0,0,0,0,0,0,0],[0,1./3.,1./3.,0,1./3.,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]]
		
		Initialize class for three quibit states
		>>> gmntool = gmn([2,2,2])

		Set GHZ state and calculate the genuine multiparticlce negativity
		>>> gmntool.setdensitymatrix(ghz)
		>>> gmntool.gmn()
		0.499999998369068
		
		Set W state and calculate the genuine multiparticlce negativity
		>>> gmntool.setdensitymatrix(w)
		>>> gmntool.gmn()
		0.4714045153813209
		"""
		self.__rho = densitymatrix(rho,self.__dimsubs)
	def setoperatorbasis(self,opbasis=[]):
		"""
		Set an custom operator basis
		If this basis does not span the full operator space the resulting witnesses lie within the subspace only

		opbasis : list of array_like
			A list of array_like objects representing a basis in the space of operators/observables.

		Try to find the optimal witness within the operator subspace spanned by pauli operators X,Y,Z and qubit identity matrices (1): 111, XXZ, XZX, ZXX, ZZZ
		>>> from gmntools import gmn, pauli
		>>> qb_3 = pauli([2,2,2])
		>>> basis = [qb_3.operator(i) for i in ['eee','xxz','xzx','zxx','zzz']]
		>>>  ghz = [[.5,0,0,0,0,0,0,.5],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[.5,0,0,0,0,0,0,.5]]
		>>> gmntool = gmn([2,2,2],ghz)

		By default a complete basis is used
		>>> gmntool.gmn()
		0.499999998369068

		There is no witness within the subspace detecting the GHZ state
		>>> gmntool.setoperatorbasis(basis)
		>>> gmntool.gmn()
		-3.646152630556347e-08

		To reset to the full standard basis call the function without argument
		>>> gmntool.setoperatorbasis()
		>>> gmntool.gmn()
		0.499999998369068
		"""
		self.operatorbasis = [hoperator(o,self.__dimsubs) for o in opbasis]
	def setrealbasis(self):
		"""
		Use real operator basis to speed up calculations

		Evaluate the genuine multiparticlce negativity first for the GHZ state and afterwards for the W state
		>>> from gmntools import gmn
		
		Set GHZ state
		>>> ghz = [[.5,0,0,0,0,0,0,.5],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[.5,0,0,0,0,0,0,.5]]
		
		Initialize class for three quibit states
		>>> gmntool = gmn([2,2,2],ghz)

		Normal evaluation
		>>> gmntool.gmn()
		0.499999998369068
		
		Fast evaluation using a real operator basis
		>>> gmntool.setrealbasis()
		>>> gmntool.gmn(real=True)
		0.4999999944514258
		"""
		self.__initrealoperatorbasis()
	def setpaulibasis(self):
		"""
		Set Pauli operator basis
		Use a operator basis consisting of tensor products of Pauli operators

		Try to find the optimal witness within the operator subspace spanned by pauli operators X,Y,Z and qubit identity matrices (1): 111, XXZ, XZX, ZXX, ZZZ
		>>> from gmntools import gmn
		>>>  ghz = [[.5,0,0,0,0,0,0,.5],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[.5,0,0,0,0,0,0,.5]]
		>>> gmntool = gmn([2,2,2],ghz)

		Use a basis of tensor products of Pauli matrices as basis
		>>> gmntool.setpaulibasis()
		>>> gmntool.gmn()
		0.49999999836906783

		Reset to the default operator basis
		>>> gmntool.setoperatorbasis()
		>>> gmntool.gmn()
		0.499999998369068
		"""
		if not all([2==i for i in self.__dimsubs]):
			raise TypeError("Memberfuction availible in qubit systems only.")
		self.setoperatorbasis(pauli(self.__nsys).basis())
	def setsympaulibasis(self,rho=None):
		"""
		Set symmetric Pauli operator basis
		Determine internal symmetries of a state to provide a basis spanning the minimal subspace containing the optimal witness

		rho : array_like
			An array_like object representing a density matrix of the given system.

		As an example use the basis symmetric with respect to the GHZ and W state to calculate the genuine multiparticle negativity of the GHZ state
		>>> from gmntools import gmn
		>>>  ghz = [[.5,0,0,0,0,0,0,.5],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[.5,0,0,0,0,0,0,.5]]
		>>> w = [[0,0,0,0,0,0,0,0],[0,1./3.,1./3.,0,1./3.,0,0,0],[0,1./3.,1./3.,0,1./3.,0,0,0],[0,0,0,0,0,0,0,0],[0,1./3.,1./3.,0,1./3.,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]]
		>>> gmntool = gmn([2,2,2],ghz)

		Determine a basis symmetric with respect to the current density matrix
		>>> gmntool.setsympaulibasis()
		>>> gmntool.gmn()
		0.4999999983690673

		Change to a basis symmetric with respect to the W state. No witness detecting genuine multiparticle entangelement is found within this subspace
		>>> gmntool.setsympaulibasis(w)
		>>> gmntool.gmn()
		-1.7734501447089677e-09
		"""
		if not all([2==i for i in self.__dimsubs]):
			raise TypeError("Memberfuction availible in qubit systems only.")
		if not rho:
			if not self.__rho:
				raise TypeError("Set density matrix before calling Memberfunction or provide density matrix as argument.")
			else:
				self.setoperatorbasis(pauli(self.__nsys).symbasis(self.__rho.matrix))
		else:
			self.setoperatorbasis(pauli(self.__nsys).symbasis(rho))
	def __F0toh(self,F0,real=False):
		if not real:
			return [cvxopt.matrix(numpy.bmat([[op.real, -op.imag], [op.imag, op.real]]),tc='d') for op in F0]
		else:
			return [cvxopt.matrix(op.real.tolist(), tc='d') for op in F0]
	def __FtoG(self,F,real=False):
		if not real:
			return [cvxopt.sparse(cvxopt.matrix([[i for i in numpy.bmat([[-Fi.real, Fi.imag], [-Fi.imag, -Fi.real]]).flat] for Fi in line],tc='d'),tc='d') for line in F]
		else:
			return [cvxopt.sparse(cvxopt.matrix([[i for i in (-Gi).real.flat] for Gi in line],tc='d'),tc='d') for line in F]
	def __solve(self,c,F0,F,real=False,altsolver=None):
		c = cvxopt.matrix(c,tc='d')
		h = self.__F0toh(F0,real)
		G = self.__FtoG(F,real)
		if not altsolver:
			return solvers.sdp(c,Gs=G,hs=h)
		else:
			return solvers.sdp(c,Gs=G,hs=h,solver='dsdp')
	def __W(self,sol):
		nop = len(self.operatorbasis)
		self.W = {}
		W =  numpy.sum([sol['x'][index]*op.matrix for index,op in enumerate(self.operatorbasis)],axis=0)
		self.W['W'] = W
		for i in range(1,2**(self.__nsys -1)):
			temp = map(int,numpy.binary_repr(i,self.__nsys))
			subsys = []
			for index,j in enumerate(temp):
				if j ==1:
					subsys.append(index)
			m = (str(subsys)+'|'+str([l for l in range(1,self.__nsys+1) if l not in subsys])).replace('[','').replace(']','').replace(' ','')
			Pm = numpy.sum([sol['x'][i*nop+k]*op.matrix for k,op in enumerate(self.operatorbasis)],axis=0)
			self.W['P_'+m] = Pm
			self.W['Q_'+m] = hoperator(W-Pm,self.__dimsubs).ptranspose(subsys)
		return 0
	def __rhom(self,sol,real=False):
		self.rhom = {}
		for i in range(1,2**(self.__nsys -1)):
			temp = map(int,numpy.binary_repr(i,self.__nsys))
			subsys = []
			for index,j in enumerate(temp):
				if j ==1:
					subsys.append(index)
			m = (str(subsys)+'|'+str([l for l in range(1,self.__nsys+1) if l not in subsys])).replace('[','').replace(']','').replace(' ','')
			rhom = zeros(self.__dim, dtype=numpy.complex128)
			if not real:
				rhom += 2*array(sol['zs'][i-1])[:self.__dimH,:self.__dimH]
				rhom += 2.j*array(sol['zs'][i-1])[self.__dimH:,:self.__dimH]
			else:
				rhom += array(sol['zs'][i-1])
			self.rhom['rho_'+m] = rhom
		return 0
	def gmn(self,real=False,altsolver=None):
		if not self.__rho:
			raise Exception("Memberfuction gmn() can not be used prior to setting density matrix using the Memberfuction setdensitymatrix(rho).")
		rho = self.__rho.matrix
		#initialize standard operatorbasis if no operatorbasis is provided
		if not self.operatorbasis:
			self.__initoperatorbasis()
		#setting up SPD
		##setting up problem vector
		nop = len(self.operatorbasis)
		c = zeros(nop*2**(self.__nsys-1), dtype=numpy.float64)
		for index,o in enumerate(self.operatorbasis):
			c[index] = numpy.trace(numpy.dot(rho,o.matrix)).real
		##setting up semidefinite constraints
		F0 = []
		F = []
		##setting up constraint P_m >= 0
		for i in range(1,2**(self.__nsys-1)):
			F0 += [zeros(self.__dim)]
			F += [[zeros(self.__dim) for j in range(i*nop)] + [o.matrix for o in self.operatorbasis] + [zeros(self.__dim) for j in range((2**(self.__nsys-1)-i-1)*nop)]]
		##setting up constraint (W-P_m)^(T_m) >= 0
		for i in range(1,2**(self.__nsys -1)):
			temp = map(int,numpy.binary_repr(i,self.__nsys))
			subsys = []
			for index,j in enumerate(temp):
				if j ==1:
					subsys.append(index)
			F0 += [zeros(self.__dim)]
			F += [[o.ptranspose(subsys) for o in self.operatorbasis] + [zeros(self.__dim) for j in range((i-1)*nop)] + [-o.ptranspose(subsys) for o in self.operatorbasis] + [zeros(self.__dim) for j in range((2**( self.__nsys -1)-i-1)*nop)]]
		##setting up constraint (W-P_m)^(T_m) <= 1
		for i in range(1,2**( self.__nsys -1)):
			temp = map(int,numpy.binary_repr(i,self.__nsys))
			subsys = []
			for index,j in enumerate(temp):
				if j ==1:
					subsys.append(index)
			F0 += [eye(self.__dimH,dtype=numpy.complex128)]
			F += [[-o.ptranspose(subsys) for o in self.operatorbasis] + [zeros(self.__dim) for j in range((i-1)*nop)] + [o.ptranspose(subsys) for o in self.operatorbasis] + [zeros(self.__dim) for j in range((2**( self.__nsys -1)-i-1)*nop)]]
		sol = self.__solve(c,F0,F,real,altsolver)
		self.status = sol['status']
		self.__W(sol)
		self.__rhom(sol,real)
		return -sol['primal objective']
	def gmn_jmg(self,real=False,altsolver=None):
		if not self.__rho:
			raise Exception("Memberfuction gmn() can not be used prior to setting density matrix using the Memberfuction setdensitymatrix(rho).")
		rho = self.__rho.matrix
		#initialize standard operatorbasis if no operatorbasis is provided
		if not self.operatorbasis:
			self.__initoperatorbasis()
		#setting up SPD
		##setting up problem vector
		nop = len(self.operatorbasis)
		c = zeros(nop*2**(self.__nsys-1), dtype=numpy.float64)
		for index,o in enumerate(self.operatorbasis):
			c[index] = numpy.trace(numpy.dot(rho,o.matrix)).real
		##setting up semidefinite constraints
		F0 = []
		F = []
		##setting up constraint P_m >= 0
		for i in range(1,2**(self.__nsys-1)):
			F0 += [zeros(self.__dim)]
			F += [[zeros(self.__dim) for j in range(i*nop)] + [o.matrix for o in self.operatorbasis] + [zeros(self.__dim) for j in range((2**(self.__nsys-1)-i-1)*nop)]]
		##setting up constraints P_m <= 1
		for i in range(1,2**(self.__nsys-1)):
			F0 += [eye(self.__dimH,dtype=numpy.complex128)]
			F += [[zeros(self.__dim) for j in range(i*nop)] + [-o.matrix for o in self.operatorbasis] + [zeros(self.__dim) for j in range((2**(self.__nsys-1)-i-1)*nop)]]
		##setting up constraint (W-P_m)^(T_m) >= 0
		for i in range(1,2**(self.__nsys -1)):
			temp = map(int,numpy.binary_repr(i,self.__nsys))
			subsys = []
			for index,j in enumerate(temp):
				if j ==1:
					subsys.append(index)
			F0 += [zeros(self.__dim)]
			F += [[o.ptranspose(subsys) for o in self.operatorbasis] + [zeros(self.__dim) for j in range((i-1)*nop)] + [-o.ptranspose(subsys) for o in self.operatorbasis] + [zeros(self.__dim) for j in range((2**( self.__nsys -1)-i-1)*nop)]]
		##setting up constraint (W-P_m)^(T_m) <= 1
		for i in range(1,2**( self.__nsys -1)):
			temp = map(int,numpy.binary_repr(i,self.__nsys))
			subsys = []
			for index,j in enumerate(temp):
				if j ==1:
					subsys.append(index)
			F0 += [eye(self.__dimH,dtype=numpy.complex128)]
			F += [[-o.ptranspose(subsys) for o in self.operatorbasis] + [zeros(self.__dim) for j in range((i-1)*nop)] + [o.ptranspose(subsys) for o in self.operatorbasis] + [zeros(self.__dim) for j in range((2**( self.__nsys -1)-i-1)*nop)]]
		sol = self.__solve(c,F0,F,real,altsolver)
		self.status = sol['status']
		self.__W(sol)
		return -sol['primal objective']
	def __rho_part_info(self,sol,real= False):
		rho = zeros(self.__dim, dtype=numpy.complex128)
		if not real:
			rho += 2*array(sol['zs'][0])[:self.__dimH,:self.__dimH]
			rho += 2.j*array(sol['zs'][0])[self.__dimH:,:self.__dimH]
		else:
			rho += array(sol['zs'][0])
		tmp = (rho+rho.conjugate().transpose())/2.
		tmp /= numpy.trace(tmp)
		self.setdensitymatrix(tmp)
		return 0
	def __W_part_info(self,sol,measurements):
		nmes= len(measurements)
		for i in range(nmes):
			self.W_part_info += sol['x'][i]*measurements[i][0]
		return 0
	def gmn_partial_info(self,meas,real=False,altsolver=None):
		measurements = list(meas)
		if type(measurements) is not list:
			if [m for m in measurements if (type(m) is not tuple and type(m) is not list) or len(m)!=2 or type(m[1]) not in [int,complex,float,long]]:
				raise TypeError("'mesurements' must be a list of tuples containing the measured operator and its expectation value '(operator,expectation_value)'")
		measurements += [(eye(self.__dimH,dtype=numpy.complex128),1.)]
		#initialize standard operatorbasis if no operatorbasis is provided
		if not self.operatorbasis:
			self.__initoperatorbasis()
		#setting up SDP
		##setting up problem vector
		nmes= len(measurements)
		mesop = [array(m[0], dtype=numpy.complex128) for m in measurements]
		nop = len(self.operatorbasis)
		c = zeros(nmes+nop*2**(self.__nsys-1), dtype=numpy.float64)
		for index,o in enumerate(measurements):
			c[index] = o[1]
		##setting up semidefinite constraints
		F0 = []
		F = []
		##setting up constraint M >= W
		F0 += [zeros(self.__dim)]
		F += [[m[0] for m in measurements] + [-o.matrix for o in self.operatorbasis] + [zeros(self.__dim) for j in range((2**(self.__nsys-1)-1)*nop)]]
		##setting up constraint Q_m <= 1
		for i in range(1,2**(self.__nsys-1)):
			F0 += [eye(self.__dimH,dtype=numpy.complex128)]
			F += [[zeros(self.__dim) for j in range(nmes)] + [zeros(self.__dim) for j in range(i*nop)] + [-o.matrix for o in self.operatorbasis] + [zeros(self.__dim) for j in range((2**(self.__nsys-1)-i-1)*nop)]]
		##setting up constraint Q_m >= 0
		for i in range(1,2**(self.__nsys-1)):
			F0 += [zeros(self.__dim)]
			F += [[zeros(self.__dim) for j in range(nmes)] + [zeros(self.__dim) for j in range(i*nop)] + [o.matrix for o in self.operatorbasis] + [zeros(self.__dim) for j in range((2**(self.__nsys-1)-i-1)*nop)]]
		##setting up constraint W-Q_m^(T_m) >= 0
		for i in range(1,2**(self.__nsys -1)):
			temp = map(int,numpy.binary_repr(i,self.__nsys))
			subsys = []
			for index,j in enumerate(temp):
				if j ==1:
					subsys.append(index)
			F0 += [zeros(self.__dim)]
			F += [[zeros(self.__dim) for j in range(nmes)] + [o.matrix for o in self.operatorbasis] + [zeros(self.__dim) for j in range((i-1)*nop)] + [-o.ptranspose(subsys) for o in self.operatorbasis] + [zeros(self.__dim) for j in range((2**( self.__nsys -1)-i-1)*nop)]]
		sol = self.__solve(c,F0,F,real,altsolver)
		self.status = sol['status']
		self.__rho_part_info(sol,real)
		self.__W_part_info(sol,measurements)
		return -sol['primal objective']
	def gmn_partial_info_ppt(self,meas,real=False,altsolver=None):
		measurements = list(meas)
		if type(measurements) is not list:
			if [m for m in measurements if (type(m) is not tuple and type(m) is not list) or len(m)!=2 or type(m[1]) not in [int,complex,float,long]]:
				raise TypeError("'mesurements' must be a list of tuples containing the measured operator and its expectation value '(operator,expectation_value)'")
		measurements += [(eye(self.__dimH,dtype=numpy.complex128),1.)]
		#setting up SDP
		##setting up problem vector
		nmes= len(measurements)
		mesop = [hoperator(m[0],self.__dimsubs) for m in measurements]
		c = zeros(nmes, dtype=numpy.float64)
		for index,o in enumerate(measurements):
			c[index] = o[1]
		##setting up semidefinite constraints
		F0 = []
		F = []
		##setting up constraint W^(T_m) >= 0
		for i in range(1,2**(self.__nsys -1)):
			temp = map(int,numpy.binary_repr(i,self.__nsys))
			subsys = []
			for index,j in enumerate(temp):
				if j ==1:
					subsys.append(index)
			F0 += [zeros(self.__dim)]
			F += [[o.ptranspose(subsys) for o in mesop]]
		##setting up constraint W^(T_m) <= 1
		for i in range(1,2**(self.__nsys -1)):
			temp = map(int,numpy.binary_repr(i,self.__nsys))
			subsys = []
			for index,j in enumerate(temp):
				if j ==1:
					subsys.append(index)
			F0 += [eye(self.__dimH)]
			F += [[-o.ptranspose(subsys) for o in mesop]]
		sol = self.__solve(c,F0,F,real,altsolver)
		self.status = sol['status']
		return -sol['primal objective']

# End of gmntools.py
