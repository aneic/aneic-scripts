'''
Generate html file of results
'''
import os
import pickle
import gzip
import numpy as np
import pandas as pd

def write_csv(filename, gamma):
    N, K = gamma.shape
    data = pd.concat([pd.DataFrame([k * np.ones(N), gamma[k].values], 
                                   columns=gamma.index, 
                                   index=['Cluster %d' % k, 'Probability%d' % k]).T 
                      for k in  gamma], axis=1)
    data.to_csv(open(filename, 'w'))

if __name__ == '__main__':
    # parse command-line args
    from optparse import OptionParser
    usage = "usage: %prog [options] files"
    parser = OptionParser(usage=usage)
    parser.add_option('-d', '--dir',
                      metavar="DIR", help="output directory", dest='dir')
    (opt, args) = parser.parse_args()

    # find files matching pattern for each supplied argument
    import glob
    files = []
    for pattern in args:
        files += glob.glob(pattern)

    # create output dir if necessary
    import os
    if opt.dir and files:
        out_dir = opt.dir
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
    else:       
        out_dir = os.getcwd()
    
    # write out csv 
    data = None
    for fname in files:
        # load responsibilities
        gamma = pickle.load(gzip.open(fname, 'r'))['gamma']
        # determine output filename
        csv_name = '%s/%s' % (out_dir, fname.replace('.pk.gz', '.csv').split('/')[-1])
        # write output
        write_csv(csv_name, gamma)
