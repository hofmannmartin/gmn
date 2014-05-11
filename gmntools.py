#!/usr/bin/python
# Filename: gmntools.py

import gmnutils
from gmnutils import densitymatrix, hoperator, pauli

import cvxopt
from cvxopt.base import spmatrix
from cvxopt import solvers

import numpy
from numpy import array, arange, prod, zeros

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
			#print 'Initializing gmn class with given densitymatrix'
			self.setdensitymatrix(matrix)
		else:
			#print 'initializing gmn class with fully mixed state'
			self.__rho = None
		#self.operatorbasis = [hoperator(o,subsystems) for o in opbasis]
		self.setoperatorbasis(opbasis)
	def __initoperatorbasis(self):
		#print 'initializing standard operator basis ...'
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
		#print 'done'
	def __initrealoperatorbasis(self):
		#print 'initializing standard operator basis ...'
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
		#print 'done'
	def witness_expectationvalue(self,rho=None):
		if not self.W:
			raise TypeError("Memberfuction availible after using Memberfunctin gmn only.")
		if not rho:
			return numpy.trace(numpy.dot(self.W['W']),self.__rho.matrix)
		else:
			return numpy.trace(numpy.dot(self.W['W']),rho)
	def setdensitymatrix(self,rho):
		#print 'setting up density matrix'
		self.__rho = densitymatrix(rho,self.__dimsubs)
	def setoperatorbasis(self,opbasis=[]):
		#print 'setting up operator basis'
		self.operatorbasis = [hoperator(o,self.__dimsubs) for o in opbasis]
	def setrealsymmbasis(self):
		#print 'setting up operator basis'
		self.__initrealoperatorbasis()
	def setpaulibasis(self):
		if not all([2==i for i in self.__dimsubs]):
			raise TypeError("Memberfuction availible in qubit systems only.")
		self.setoperatorbasis(pauli(self.__nsys).basis())
	def setsympaulibasis(self,rho=None):
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
		#print 'obtaining the witness and its decompositions from the solution...'
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
		#print 'done'
		return 0
	def __rhom(self,sol,real=False):
		#print 'obatining the decomposition of the state into \sum_m \\rho_m...'
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
		#print 'done'
		return 0
	def gmn(self,real=False,altsolver=None):
		if not self.__rho:
			raise Exception("Memberfuction gmn() can not be used prior to setting density matrix using the Memberfuction setdensitymatrix(rho).")
		rho = self.__rho.matrix
		#initialize standard operatorbasis if no operatorbasis is provided
		if not self.operatorbasis:
			self.__initoperatorbasis()
		#setting up SPD
		#print 'starting to parse the semidefinite problem ...'
		#setting up problem vector
		#print '\tinititializing problem vector c ...'
		nop = len(self.operatorbasis)
		c = numpy.zeros(nop*2**(self.__nsys-1), dtype=numpy.float64)
		for index,o in enumerate(self.operatorbasis):
			c[index] = numpy.trace(numpy.dot(rho,o.matrix)).real
		#print '\tdone'
		#setting up semidefinite constraints
		#print '\tinitializing semidefinite constraints ...'
		F0 = []
		F = []
		##setting up constraint P_m >= 0
		#print '\t\tsetting up P_m >= 0 ...'
		for i in range(1,2**(self.__nsys-1)):
			F0 += [numpy.zeros(self.__dim)]
			F += [[numpy.zeros(self.__dim) for j in range(i*nop)] + [o.matrix for o in self.operatorbasis] + [numpy.zeros(self.__dim) for j in range((2**(self.__nsys-1)-i-1)*nop)]]
		#print '\t\tdone'
		##setting up constraint (W-P_m)^(T_m) >= 0
		#print '\t\tsetting up Q_m >= 0 ...'
		for i in range(1,2**(self.__nsys -1)):
			temp = map(int,numpy.binary_repr(i,self.__nsys))
			subsys = []
			for index,j in enumerate(temp):
				if j ==1:
					subsys.append(index)
			F0 += [numpy.zeros(self.__dim)]
			F += [[o.ptranspose(subsys) for o in self.operatorbasis] + [numpy.zeros(self.__dim) for j in range((i-1)*nop)] + [-o.ptranspose(subsys) for o in self.operatorbasis] + [numpy.zeros(self.__dim) for j in range((2**( self.__nsys -1)-i-1)*nop)]]
		#print '\t\tdone'
		##setting up constraint (W-P_m)^(T_m) <= 1
		#print '\t\tsetting up Q_m <= 1 ...'
		for i in range(1,2**( self.__nsys -1)):
			temp = map(int,numpy.binary_repr(i,self.__nsys))
			subsys = []
			for index,j in enumerate(temp):
				if j ==1:
					subsys.append(index)
			F0 += [numpy.eye(self.__dimH,dtype=numpy.complex128)]
			F += [[-o.ptranspose(subsys) for o in self.operatorbasis] + [numpy.zeros(self.__dim) for j in range((i-1)*nop)] + [o.ptranspose(subsys) for o in self.operatorbasis] + [numpy.zeros(self.__dim) for j in range((2**( self.__nsys -1)-i-1)*nop)]]
		#print '\t\tdone'
		#print '\tdone'
		#print 'done'
		#print 'solving semidefinite program...'
		sol = self.__solve(c,F0,F,real,altsolver)
		#print 'done'
		self.status = sol['status']
		self.__W(sol)
		self.__rhom(sol,real)
		return -sol['primal objective']
	def __rho_part_info(self,sol,real= False):
		#print 'obatining the least entangled state compatible with measurements...'
		rho = zeros(self.__dim, dtype=numpy.complex128)
		if not real:
			rho += 2*array(sol['zs'][0])[:self.__dimH,:self.__dimH]
			rho += 2.j*array(sol['zs'][0])[self.__dimH:,:self.__dimH]
		else:
			rho += array(sol['zs'][0])
		tmp = (rho+rho.conjugate().transpose())/2.
		tmp /= numpy.trace(tmp)
		self.setdensitymatrix(tmp)
		#print 'done'
		return 0
	def __W_part_info(self,sol,measurements):
		#print 'obatining the optimal witness within the span of all measured operators...'
		nmes= len(measurements)
		for i in range(nmes):
			self.W_part_info += sol['x'][i]*measurements[i][0]
		#print 'done'
		return 0
	def gmn_partial_info(self,meas,real=False,altsolver=None):
		measurements = list(meas)
		if type(measurements) is not list:
			if [m for m in measurements if (type(m) is not tuple and type(m) is not list) or len(m)!=2 or type(m[1]) not in [int,complex,float,long]]:
				raise TypeError("'mesurements' must be a list of tuples containing the measured operator and its expectation value '(operator,expectation_value)'")
		measurements += [(numpy.eye(self.__dimH,dtype=numpy.complex128),1.)]
		#initialize standard operatorbasis if no operatorbasis is provided
		if not self.operatorbasis:
			self.__initoperatorbasis()
		#setting up SDP
		#print 'starting to parse the semidefinite problem ...'
		#setting up problem vector
		#print '\tinititializing problem vector c ...'
		nmes= len(measurements)
		mesop = [array(m[0], dtype=numpy.complex128) for m in measurements]
		nop = len(self.operatorbasis)
		c = numpy.zeros(nmes+nop*2**(self.__nsys-1), dtype=numpy.float64)
		for index,o in enumerate(measurements):
			c[index] = o[1]
		#print '\tdone'
		#setting up semidefinite constraints
		#print '\tinitializing semidefinite constraints ...'
		F0 = []
		F = []
		##setting up constraint M >= W
		#print '\t\tsetting up M >= W ...'
		F0 += [numpy.zeros(self.__dim)]
		F += [[m[0] for m in measurements] + [-o.matrix for o in self.operatorbasis] + [numpy.zeros(self.__dim) for j in range((2**(self.__nsys-1)-1)*nop)]]
		#print '\t\tdone'
		##setting up constraint Q_m <= 1
		#print '\t\tsetting up Q_m <= 1 ...'
		for i in range(1,2**(self.__nsys-1)):
			F0 += [numpy.eye(self.__dimH,dtype=numpy.complex128)]
			F += [[numpy.zeros(self.__dim) for j in range(nmes)] + [numpy.zeros(self.__dim) for j in range(i*nop)] + [-o.matrix for o in self.operatorbasis] + [numpy.zeros(self.__dim) for j in range((2**(self.__nsys-1)-i-1)*nop)]]
		#print '\t\tdone'
		##setting up constraint Q_m >= 0
		#print '\t\tsetting up Q_m >= 0 ...'
		for i in range(1,2**(self.__nsys-1)):
			F0 += [numpy.zeros(self.__dim)]
			F += [[numpy.zeros(self.__dim) for j in range(nmes)] + [numpy.zeros(self.__dim) for j in range(i*nop)] + [o.matrix for o in self.operatorbasis] + [numpy.zeros(self.__dim) for j in range((2**(self.__nsys-1)-i-1)*nop)]]
		#print '\t\tdone'
		##setting up constraint W-Q_m^(T_m) >= 0
		#print '\t\tsetting up W-Q_m^(T_m) >= 0 ...'
		for i in range(1,2**(self.__nsys -1)):
			temp = map(int,numpy.binary_repr(i,self.__nsys))
			subsys = []
			for index,j in enumerate(temp):
				if j ==1:
					subsys.append(index)
			F0 += [numpy.zeros(self.__dim)]
			F += [[numpy.zeros(self.__dim) for j in range(nmes)] + [o.matrix for o in self.operatorbasis] + [numpy.zeros(self.__dim) for j in range((i-1)*nop)] + [-o.ptranspose(subsys) for o in self.operatorbasis] + [numpy.zeros(self.__dim) for j in range((2**( self.__nsys -1)-i-1)*nop)]]
		#print '\t\tdone'
		#print '\tdone'
		#print 'done'
		#print 'solving semidefinite program...'
		sol = self.__solve(c,F0,F,real,altsolver)
		#print 'done'
		self.status = sol['status']
		self.__rho_part_info(sol,real)
		self.__W_part_info(sol,measurements)
		return -sol['primal objective']

# End of mymodule.py
