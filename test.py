#!/usr/bin/python
# Filename: test.py

import gmntools, gmnutils
from gmntools import gmn
from gmntools import pauli
import time, numpy

#test0 = gmntools.gmn([2,2],[[1./2,0,0,1./2],[0,0,0,0],[0,0,0,0],[1./2,0,0,1./2]])
#print test0.gmn()

print '\nuse 3 qubit GHZ state'
test1 = gmn([2,2,2])
test1.setdensitymatrix([[1./2,0,0,0,0,0,0,1./2],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[1./2,0,0,0,0,0,0,1./2]])
print 'use standard operator basis and standard solver'
w = time.time()
print test1.gmn()
print time.time()-w,'s'

print 'use standard operator basis and dsdp solver'
w = time.time()
print test1.gmn(altsolver=True)
print time.time()-w,'s'

print 'use Pauli operator basis and standard solver'
test1.setpaulibasis()
w = time.time()
print test1.gmn()
print time.time()-w,'s'

print 'use Pauli operator basis and dsdp solver'
w = time.time()
print test1.gmn(altsolver=True)
print time.time()-w,'s'

print 'use symmetric Pauli operator basis and standard solver'
test1.setsympaulibasis()
w = time.time()
print test1.gmn()
print time.time()-w,'s'

print 'use symmetric Pauli operator basis and dsdp solver'
w = time.time()
print test1.gmn(altsolver=True)
print time.time()-w,'s'

print 'use symmetric real Pauli operator basis and standard solver'
w = time.time()
print test1.gmn(real=True)
print time.time()-w,'s'

print 'use symmetric real Pauli operator basis and dsdp solver'
w = time.time()
print test1.gmn(real=True,altsolver=True)
print time.time()-w,'s'

print '\nuse 4 qubit GHZ state'
rho = numpy.zeros((16,16))
rho[0,0] = rho[0,-1] = rho[-1,0] = rho[-1,-1] = .5
test2 = gmn([2,2,2,2],rho)
#print 'use standard operator basis and standard solver'
#w = time.time()
#print test2.gmn()
#print time.time()-w,'s'

#print 'use standard operator basis and dsdp solver'
#w = time.time()
#print test2.gmn(altsolver=True)
#print time.time()-w,'s'

#print 'use Pauli operator basis and standard solver'
#test2.setpaulibasis()
#w = time.time()
#print test2.gmn()
#print time.time()-w,'s'

#print 'use Pauli operator basis and dsdp solver'
#w = time.time()
#print test2.gmn(altsolver=True)
#print time.time()-w,'s'

print 'use symmetric Pauli operator basis and standard solver'
test2.setsympaulibasis()
w = time.time()
print test2.gmn()
print time.time()-w,'s'

print 'use symmetric Pauli operator basis and dsdp solver'
w = time.time()
print test2.gmn(altsolver=True)
print time.time()-w,'s'

print 'use symmetric real Pauli operator basis and standard solver'
w = time.time()
print test2.gmn(real=True)
print time.time()-w,'s'

print 'use symmetric real Pauli operator basis and dsdp solver'
w = time.time()
print test2.gmn(real=True,altsolver=True)
print time.time()-w,'s'

#print '\nuse 5 qubit GHZ state'
#rho = numpy.zeros((32,32))
#rho[0,0] = rho[0,-1] = rho[-1,0] = rho[-1,-1] = .5
#test3 = gmn([2,2,2,2,2],rho)
#print 'use symmetric real Pauli operator basis and dsdp solver'
#test3.setsympaulibasis()
#w = time.time()
#print test3.gmn(real=True,altsolver=True)
#print time.time()-w,'s'

#print '\nuse 6 qubit GHZ state'
#rho = numpy.zeros((64,64))
#rho[0,0] = rho[0,-1] = rho[-1,0] = rho[-1,-1] = .5
#test4 = gmn([2,2,2,2,2,2],rho)
#print 'use symmetric real Pauli operator basis and dsdp solver'
#test4.setsympaulibasis()
#w = time.time()
#print test4.gmn(real=True,altsolver=True)
#print time.time()-w,'s'

def Dicke(n,k):
	dickestate = numpy.zeros(2**n, dtype=numpy.complex128)
	for i in range(2**n):
		if numpy.binary_repr(i).count('1') == k:
			dickestate[i] = 1.
	rho = numpy.outer(dickestate,dickestate)
	return rho/numpy.trace(rho)

print '\nchange to Dicke (4,2) state'
D42 = Dicke(4,2)
test2.setdensitymatrix(D42)
test2.setoperatorbasis()
print 'use standard operator basis and dsdp solver'
w = time.time()
print test2.gmn(altsolver=True)
print time.time()-w,'s'

test2.setrealsymmbasis()
print 'use real standard operator basis and dsdp solver'
w = time.time()
print test2.gmn(real=True,altsolver=True)
print time.time()-w,'s'


print 'use symmetric Pauli operator basis and dsdp solver'
test2.setsympaulibasis()
w = time.time()
print test2.gmn(real=True,altsolver=True)
print time.time()-w,'s'



#import numpy
#from numpy import array

#rho = array([[1./2,0,0,0,0,0,0,1./2],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[1./2,0,0,0,0,0,0,1./2]])

#ZEZ = array([[1,0,0,0,0,0,0,0],[0,-1.,0,0,0,0,0,0],[0,0,1.,0,0,0,0,0],[0,0,0,-1.,0,0,0,0],[0,0,0,0,-1.,0,0,0],[0,0,0,0,0,1.,0,0],[0,0,0,0,0,0,-1.,0],[0,0,0,0,0,0,0,1.]])

#EZZ = array([[1,0,0,0,0,0,0,0],[0,-1.,0,0,0,0,0,0],[0,0,-1.,0,0,0,0,0],[0,0,0,1.,0,0,0,0],[0,0,0,0,1.,0,0,0],[0,0,0,0,0,-1.,0,0],[0,0,0,0,0,0,-1.,0],[0,0,0,0,0,0,0,1.]])

#ZZE = array([[1,0,0,0,0,0,0,0],[0,1.,0,0,0,0,0,0],[0,0,-1.,0,0,0,0,0],[0,0,0,-1.,0,0,0,0],[0,0,0,0,-1.,0,0,0],[0,0,0,0,0,-1.,0,0],[0,0,0,0,0,0,1.,0],[0,0,0,0,0,0,0,1.]])

#XXX = array([[0,0,0,0,0,0,0,1.],[0,0,0,0,0,0,1.,0],[0,0,0,0,0,1.,0,0],[0,0,0,0,1.,0,0,0],[0,0,0,1.,0,0,0,0],[0,0,1.,0,0,0,0,0],[0,1.,0,0,0,0,0,0],[1.,0,0,0,0,0,0,0]])

#measurements = [(EZZ,numpy.trace(numpy.dot(rho,EZZ))),(ZEZ,numpy.trace(numpy.dot(rho,ZEZ))),(ZZE,numpy.trace(numpy.dot(rho,ZZE))),(XXX,numpy.trace(numpy.dot(rho,XXX)))]

#test2 = gmntools.gmn([2,2,2])
#print test2.gmn_partial_info([])
#print test2.gmn_partial_info(measurements)
		
# End of mymodule.py
