# File: incompletemeas_multipart.py
# By Martin Hofmann
# 20.06.2014
#
# In this script randomly generated (equally distributed wrt the Haar measure)
# states are taken and their genuine multipartice negativity (GNM) is calculated.
# Furthermore, genuiune multiparticle entanglement is detected via an
# incomplete set of measurements (with a minimal number of N measurements).
# The output is given by 3 Numbers: GNM, NMIN, BOUND
# where GNM is the genuine multipartice negativity of the random state, NMIN is
# the minimal number of measurements necessary to detect genuine multiparticle 
# entanglement in that state and BOUND is the lower bound on the GNM that
# can be achieved using NMIN measurements.

import gmntools, numpy
from gmntools import gmn, randomstates, randomlocalmeas

#number of runs
n = 100
# 2x2x2 system
subsystems = [2,2,2]

# initialize necesarry classes
locmeas = randomlocalmeas(subsystems)
rho = randomstates(subsystems)
GMN = gmn(subsystems)
dim = numpy.prod(numpy.array(subsystems))

for i in range(n):
	GMN.setdensitymatrix(rho.random())
	genneg = GMN.gmn(altsolver='dsdp')
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
