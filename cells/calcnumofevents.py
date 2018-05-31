import h5py as h

count = 0
for i in range(0,78):
	a = h.File("cells_Zee_{:04d}.h5".format(i))
	b = a["egamma"]
	count += len(b)
print "count ",count
