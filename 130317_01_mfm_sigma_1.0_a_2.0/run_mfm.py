import sys
import os
import pandas as pd
import numpy as np

# hard-coded path to working directory (set to full path on clusters)
work_dir = os.getcwd()

# path to aneic-core sources (assumed to be in working directory)
ac_path = '%s/aneic-core' % work_dir

# load aneic.mfm module
sys.path.append('%s/src' % ac_path)
import aneic.mfm as mfm

if __name__ == '__main__':
    # edit diz plz
    opts = {
        # run series
        'series' : '130317_01',
        # prefix for saving file
        'outdir' : '%s/results' % work_dir,
        # dataset to load from
        'filename' : '%s/data/levantine_corpus_gansell.csv' % ac_path,
        # any strings to match to empty field
        'na_values' : ['', ' '],
        # columns to ignore
        'drop_feats' : ['Site'],
        # rows to ignore
        'drop_objs' : [],
        # number of clusters
       'K' : [2,3,4,5,6,7,8,9,10],
#       'K' : [3,],
        # maximum number of iterations
        'max_iter' : 1000,
        # convergence threshold
        'eps' : 1e-3,
        # min / max number of restarts to perform
       'min_restarts' : 100,
       'max_restarts' : 500,
#        'min_restarts' : 10,
#        'max_restarts' : 100,
        # convergence threshold for restarts
        'eps_kl' : 0.1,
        # (a-1) sets the number of pseudocounts of the prior on l 
        'a' : 2.0,
        # b[ft] = sigma**2 * var(real[ft]) * (a-1)
        # i.e. the mode of the prior Gamma(l[ft] | a, b[ft]) 
        # is 1 / sigma**2 * var(real[ft])
        'sigma' : 1.0,
        # prior on rho (set to 1 or None for max likelihood updates)  
#        'alpha' : 1 + 1e-6,
        'alpha' : None,
        # prior on z (set to 1 or None for max likelihood updates) 
#        'beta' : 1 + 1e-6 ,
        'beta' : None,
        'interval' : 10,
        # random number seed
        'seed' : 1,
        }

    # load data
    data = pd.read_csv('%s' %(opts['filename']), index_col=0, 
                       na_values=opts['na_values'])\
               .drop(opts['drop_feats'], axis=1)\
               .drop(opts['drop_objs'], axis=0)
    # split real and cat feats
    real = data[data.columns[data.dtypes == np.float64]]
    cat = mfm.convert_indicator(data[data.columns[data.dtypes == np.object]])
    # number of examples
    N = data.shape[0]

    # set opts to return None for un-itialized items
    from collections import defaultdict
    opts = defaultdict(lambda: None, opts)

    for K in opts['K']:
        # initialize prior parameters
        u = defaultdict(lambda: None)
        if not opts['a'] is None:
            u['a'] = opts['a']
            u['b'] = pd.concat([(u['a']-1) * opts['sigma']**2 
                                * real.var(0)]*K, axis=1)
        if not opts['alpha'] is None:
            u['alpha'] = pd.DataFrame(opts['alpha'] * np.ones((cat.shape[1], K)), index=cat.columns)
        if not opts['beta'] is None:
            u['z'] = pd.Series(opts['beta'] * np.ones((K,)))
        
        # do restarts
        second = None
        best = {'opts': dict(opts), 'u': dict(u), 'real': real, 'cat': cat}
        for r in range(opts['max_restarts']):
            q, g, L, g0 = mfm.em(real, cat, K, u, opts['eps'], opts['max_iter'])
            updated = False
            if r==0 or ((len(L)>1) and (L[-1] > best['L'][-1])):
                if (r > 0):
                    second = best.copy()
                best.update({'theta': q, 'gamma': g, 'L': L, 'gamma0': g0, 'restart': r})
                updated = True
            elif (not second is None) and (L[-1] > second['L'][-1]):
                second.update({'theta': q, 'gamma': g, 'L': L, 'gamma0': g0, 'restart': r})
                updated = True

            if updated: 
                if not second is None:
                    kl = mfm.kl_gamma(best['gamma'], second['gamma'])
                else:
                    kl = np.nan
                if best['L'][-1] == L[-1]:
                    print 'K: %02d  r: %03d  it: %03d  L:  (%.4e)   kl: %.4e' % (K, r, len(L), L[-1], kl)
                else:
                    print 'K: %02d  r: %03d  it: %03d  L: ((%.4e))  kl: %.4e' % (K, r, len(L), L[-1], kl)
                if (r >= opts['min_restarts']) and (kl < opts['eps_kl']):
                    break
            else:
                print 'K: %02d  r: %03d  it: %03d  L:   %.4e' % (K, r, len(L), L[-1])

        # save output
        import gzip
        import pickle
        if not os.path.isdir(opts['outdir']):
                os.makedirs(opts['outdir'])
        pickle.dump(best, gzip.open('%s/%s_K%02d_a%.4f_best.pk.gz' 
                            % (opts['outdir'], opts['series'], K, opts['a']), 'w'))
        if not second is None:
            pickle.dump(second, gzip.open('%s/%s_K%02d_a%.4f_second.pk.gz' 
                                % (opts['outdir'], opts['series'], K, opts['a']), 'w'))

