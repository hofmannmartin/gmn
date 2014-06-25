# File: incompletemeas_multipart_closetopure.py
# By Martin Hofmann
# 20.06.2014
#
# In this script randomly generated (equally distributed wrt the Haar measure)
# states are mixed with random pure states and their genuine multipartice 
# negativity (GNM) is calculated.

# Furthermore, genuiune multiparticle entanglement is detected via an
# incomplete set of measurements (with a minimal number of N measurements).
# The output is given by 3 Numbers: GNM, NMIN, BOUND
# where GNM is the genuine multipartice negativity of the random state, NMIN is
# the minimal number of measurements necessary to detect genuine multiparticle 
# entanglement in that state and BOUND is the lower bound on the GNM that
# can be achieved using NMIN measurements.

import gmntools, numpy
from gmntools import gmn, randomstates, randomlocalmeas, densitymatrix

#number of runs
n = 100
subsystems = [2,2]
weight = 0.4 #the maximal weight the mixed state can have in the mixture with the pure one

# initialize necesarry classes
locmeas = randomlocalmeas(subsystems)
rhomixed = randomstates(subsystems)
GMN = gmn(subsystems)
dim = numpy.prod(numpy.array(subsystems))

def randstate(minweight):
	psi = numpy.random.randn(dim) + 1.j*numpy.random.randn(dim)
	p = minweight*numpy.random.rand(1)
	rhopure = numpy.outer(psi.conjugate(),psi)
	rhopure /= numpy.trace(rhopure)
	return densitymatrix((1-p)*rhopure + p*rhomixed.random(),subsystems)

for i in range(n):
	rho = randstate(weight)
	GMN.setdensitymatrix(rho.matrix)
	genneg = rho.negativity([0])
	if genneg > 0:
		meas = []
		for j in range(dim**2):
			expvalue = numpy.trace(numpy.dot(rho.matrix,locmeas.random())).real
			meas += [(locmeas.matrix,expvalue)]
		xlow = 0
		xhigh = dim**2
		for k in range (8):
			x = (xhigh+xlow)/2
			y = GMN.gmn_partial_info(meas[:x])
			if y > 0:
				xhigh = x
			else:
				xlow = x
			if xhigh-xlow==1:
				print genneg, xhigh, GMN.gmn_partial_info(meas[:xhigh])
				break
