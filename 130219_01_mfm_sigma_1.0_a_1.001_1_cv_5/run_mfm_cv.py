import sys
import os
import pandas as pd
import numpy as np

# hard-coded path to working directory (set to full path on clusters)
work_dir = os.getcwd()

# path to aneic-core sources
ac_path = '%s/aneic-core' % work_dir

# load aneic.mfm module
sys.path.append('%s/src' % ac_path)
import aneic.mfm as mfm

if __name__ == '__main__':
    # edit diz plz
    opts = {
        # run series
        'series' : '130219_01',
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
        'a' : 1 + NU,
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
        'cv_folds' : 5,
        'cv_fold' : FOLD
        }

    # load data
    data = pd.read_csv('%s' %(opts['filename']), index_col=0, 
                       na_values=opts['na_values'])\
               .drop(opts['drop_feats'], axis=1)\
               .drop(opts['drop_objs'], axis=0)
    # split real and cat feats
    cat = mfm.convert_indicator(data[data.columns[data.dtypes == np.object]])
    real = data[data.columns[data.dtypes == np.float64]]
    # number of examples
    N = data.shape[0]

    # calculate splits for cross-validation
    np.random.seed(opts['seed'])
    f = opts['cv_fold']
    F = opts['cv_folds']
    shuffled = np.random.permutation(range(N));
    splits = np.round(np.arange(F + 1) * N / F)
    test = shuffled[splits[f]:splits[f+1]]
    train = np.array(list(set(shuffled) - set(test)))
    
    # store splits in opts
    opts['cv_splits'] = splits
    opts['cv_test'] = test
    opts['cv_train'] = train

    # select training and test data
    r_train = real.ix[opts['cv_train'], :]
    c_train = cat.ix[opts['cv_train'], :]
    r_test = real.ix[opts['cv_test'], :]
    c_test = cat.ix[opts['cv_test'], :]

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
        best = {'opts': dict(opts), 'u': dict(u), 'real': r_train, 'cat': c_train, 'r_test' : r_test, 'c_test' : c_test}
        for r in range(opts['max_restarts']):
            q, g, L, g0 = mfm.em(r_train, c_train, K, u, opts['eps'], opts['max_iter'])
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

        # calculate log likelihood 
        best['L_test'] = mfm.e_step(r_test, c_test, 
                            best['theta']['pi'], best['theta']['mu'], best['theta']['l'], best['theta']['rho'], 
                            a=u['a'], b=u['b'], alpha=u['alpha'], beta=u['z'])[1]      
        if not second is None:
            second['L_test'] = mfm.e_step(r_test, c_test, 
                                second['theta']['pi'], second['theta']['mu'], second['theta']['l'], second['theta']['rho'], 
                                a=u['a'], b=u['b'], alpha=u['alpha'], beta=u['z'])[1]

        # save output
        import gzip
        import pickle
        if not os.path.isdir(opts['outdir']):
                os.makedirs(opts['outdir'])
        pickle.dump(best, gzip.open('%s/%s_K%02d_a%.4f_fold%02d_best.pk.gz' 
                            % (opts['outdir'], opts['series'], K, opts['a'], opts['cv_fold']), 'w'))
        if not second is None:
            pickle.dump(second, gzip.open('%s/%s_K%02d_a%.4f_fold%02d_second.pk.gz' 
                                % (opts['outdir'], opts['series'], K, opts['a'], opts['cv_fold']), 'w'))

