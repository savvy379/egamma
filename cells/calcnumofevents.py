import h5py as h

samples = ["JF17", "JF35", "JF50", "Wminusmunu", "Zee"]
for s in samples:
    count = 0
    for i in range(0,400):
        try:
        	a = h.File("cells_{}_{:04d}.h5".format(s,i),'r')
        except:
            break;
    	b = a["egamma"]
    	count += len(b)
    print s,count
