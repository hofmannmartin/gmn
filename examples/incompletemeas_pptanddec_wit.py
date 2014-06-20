# File: incompletemeas_pptanddec_wit.py
# By Martin Hofmann
# 20.06.2014
#
# In this script randomly generated (equally distributed wrt the Haar measure)
# bipartite states are taken and their genuine multipartice negativity (GNM) is
# calculated. Furthermore, genuiune multiparticle entanglement is detected via an
# incomplete set of measurements (with a minimal number of N measurements).
# The output is given by 5 Numbers: GNM, NMIN, BOUND, NMIN_PPT, BOUND_PPT
# where GNM is the genuine multipartice negativity of the random state, NMIN is
# the minimal number of measurements necessary to detect genuine multiparticle 
# entanglement in that state and BOUND/BOUND_PPT is the lower bound on the GNM 
# that can be achieved using NMIN/NMIN_PPT measurements.
# The difference between (NMIN, BOUND) and (NMIN, BOUNT_PPT) is that in the first
# fully decomposable witnesses are constructed, wheras in the second one fully
# PPT witnesses are used.

import gmntools, numpy
from gmntools import gmn, randomstates, randomlocalmeas

#number of runs
n = 100
subsystems = [2,2]

# initialize necesarry classes
locmeas = randomlocalmeas(subsystems)
rho = randomstates(subsystems)
GMN = gmn(subsystems)
dim = numpy.prod(numpy.array(subsystems))

for i in range(n):
	GMN.setdensitymatrix(rho.random())
	genneg = rho.negativity([0])
	if genneg > 0:
		meas = []
		for j in range(dim**2):
			expvalue = numpy.trace(numpy.dot(rho.matrix,locmeas.random())).real
			meas += [(locmeas.matrix,expvalue)]
		xlow = 0
		xhigh = dim**2
		while True:
			x = (xhigh+xlow)/2
			y = GMN.gmn_partial_info(meas[:x])
			if y > 0:
				xhigh = x
			else:
				xlow = x
			if xhigh-xlow==1:
				break
		xlowppt = 0
		xhighppt = dim**2
		while True:
			x = (xhighppt+xlowppt)/2
			y = GMN.gmn_partial_info_ppt(meas[:x])
			if y > 0:
				xhighppt = x
			else:
				xlowppt = x
			if xhighppt-xlowppt==1:
				print genneg, xhigh, GMN.gmn_partial_info(meas[:xhigh]), xhighppt, GMN.gmn_partial_info_ppt(meas[:xhighppt])
				break
