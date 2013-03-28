import pandas as pd
import numpy as np
import pickle 
import gzip
from glob import glob
from collections import defaultdict

import sys
sys.path.append('./aneic-core/src')
import aneic.mfm as mfm

# oracle grouping as specified by amy
oracle = \
    [['BM118218_C20', 'BM118250_C19', 'BM118252_C18', 'BM118219_C16',
      'BM118251_C17', 'BM118158_C15', 'BM118159_C12'],
     ['BM8015', 'BM8007', '59.107.107_Met'],
     ['BM9760', 'BM7995', 'BM10447', '65.924Bost'],
     ['BM126667_S176', 'BM118234_S172', 'BM118233_S173', 'BM126666_S175',
      'BM118238_S174', 'BM126963_S192', 'BM118237_S193', 'BM118198_S197',
      'BM118228_U6_plLXXI', 'BM118217_T5', 'BM118236_S212_Fig1full',
      'BM118236_S212_Fig2frg', 'BM1 18230_S181', 'BM118193_S198', 
      'BM118203_S190', 'BM118202_S194', 'BM118181_S182', 'BM118229_U7',
      'BM118231_S213_fig1', 'BM118197_S207_fig1', 'BM118197_S207_fig2',
      'BM126675_S206_fig1Dark', 'BM126675_S206_fig2Light', 
      'BM126676_S209a-b']
    ]

# run options
opts = {
    # run series
    'series' : '130219_01',
    # prefix for saving file
    'outdir' : 'results',
    # dataset to load from
    'data' : './aneic-core/data/levantine_corpus_gansell.csv',
    # any strings to match to empty field
    'na_values' : ['', ' '],
    }

if __name__ == '__main__':
    # load data
    data = pd.read_csv('./%s' %(opts['data']), index_col=0, 
                       na_values=opts['na_values'])
    site = mfm.convert_indicator(pd.DataFrame(data['Site'], columns=['Site']))
    data = data.drop(['Site'], axis=1)
    # split real and cat feats
    cat = mfm.convert_indicator(data[data.columns[data.dtypes == np.object]])
    real = data[data.columns[data.dtypes == np.float64]]
    # number of examples
    N = data.shape[0]

    # run results
    best = defaultdict(lambda: defaultdict(lambda: {}))
    site_counts = defaultdict(lambda: defaultdict(lambda: {}))
    L = defaultdict(lambda: defaultdict(lambda: 0.0))
    Lh = defaultdict(lambda: defaultdict(lambda: 0.0))
    l = defaultdict(lambda: defaultdict(lambda: 0.0))
    lh = defaultdict(lambda: defaultdict(lambda: 0.0))
    kl_g = defaultdict(lambda: defaultdict(lambda: 0.0))
    d_g = defaultdict(lambda: defaultdict(lambda: 0.0))
    e_g = defaultdict(lambda: defaultdict(lambda: 0.0))

    for f in glob('results/*best*.pk.gz'):
        K = np.int(f.split('_')[2][1:])
        a = np.float(f.split('_')[3][1:])
        fold = np.int(f.split('_')[4][4:])
        best[a][K][fold] = pickle.load(gzip.open(f))
        L[a][K] += best[a][K][fold]['L'][-1]
        Lh[a][K] += best[a][K][fold]['L_test']

        # log joint of held out data
        u = best[a][K][fold]['u']
        theta = best[a][K][fold]['theta']
        real = best[a][K][fold]['real']
        cat = best[a][K][fold]['cat']
        l[a][K] += best[a][K][fold]['L'][-1] \
                        - mfm._log_pq(theta['mu'], theta['l'], theta['rho'], theta['pi'],
                                        u['a'], u['b']).sum()
        lh[a][K] += best[a][K][fold]['L_test'] \
                        - mfm._log_pq(theta['mu'], theta['l'], theta['rho'], theta['pi'],
                                        u['a'], u['b']).sum()

        # agreement with oracle
        gamma = best[a][K][fold]['gamma']
        flat = lambda iterable: [i for it in iterable for i in it]
        o_index = pd.Index([o for o in flat(oracle) if o in gamma.index])
        o_z = np.array([z for z,o_clust in enumerate(oracle) for o in o_clust if o in gamma.index])
        o_gamma = pd.DataFrame((o_z[:,None]==np.arange(len(oracle))[None,:]).astype('i'), index=o_index)
        kl_g[a][K] += mfm.kl_gamma(gamma, o_gamma, eps=1)
        d_g[a][K] += mfm.d_gamma(gamma, o_gamma)
        e_g[a][K] += mfm.err_gamma(gamma, o_gamma)

        # correlation of sites with clustering
        site_counts[a][K][fold] = pd.concat([(gamma[k].T * site.T).sum(1) for k in gamma], axis=1)


    c_NA = defaultdict(lambda: defaultdict(lambda: 0.0))
    c_AK = defaultdict(lambda: defaultdict(lambda: 0.0))
    c_KN = defaultdict(lambda: defaultdict(lambda: 0.0))
    for a in site_counts:
        for k in site_counts[a]:
            S = site.shape[1]
            s_counts = np.zeros((S,S,2))
            for f in site_counts[a][k]:
                # s_c[s,k] = number of observations at site s in state k
                s_c = np.array(site_counts[a][k][f])
                # p(z=k | l=s)
                pk_s = mfm.norm(s_c + 1e-6, 1).T
                # p(z1=k,z2=l | l1=s, l2=t)
                pkl_st = pk_s[:,None,:,None] * pk_s[None,:,None,:]
                # pi_st = sum_k p(z1=k, z2=k | l1=s, l2=t)
                pi_st = np.sum([pkl_st[l,l,:,:] for l in range(s_c.shape[1])], 0)
                # pi_st[:,:,1] = 1 - sum_k p(z1=k, z2=k | l1=s, l2=t)
                s_counts[:,:,0] += pi_st
                s_counts[:,:,1] += 1 - pi_st
            s_corr = pd.DataFrame(mfm.norm(s_counts, 2)[:,:,0],
                        index=site.columns.levels[1], columns=site.columns.levels[1])
            c_AK[a][k] = s_corr.ix['ArslanTash', 'Khorsabad']
            c_KN[a][k] = s_corr.ix['Khorsabad', 'Nimrud']
            c_NA[a][k] = s_corr.ix['Nimrud', 'ArslanTash']


    # L = pd.DataFrame(L)
    # Lh = pd.DataFrame(Lh)
    # lh = pd.DataFrame(lh)
    # l = pd.DataFrame(l) 
    # kl_g = pd.DataFrame(kl_g)

    # for var in ['d_g', 'e_g', 'c_AK', 'c_KN', 'c_NA']:
    #     locals()[var] = pd.DataFrame(locals()[var]) / 5

    # L = pd.concat([pd.DataFrame(L[a], index=L[a], columns=[a])
    #                for a in L], axis=1).reindex(columns=sorted(L))
    
    # Lh = pd.concat([pd.DataFrame(Lh[s], index=Lh[a], columns=[a])
    #                for a in Lh], axis=1).reindex(columns=sorted(Lh))
 
    # a = pd.concat([pd.DataFrame([best[a][k][0]['u']['a'] 
    #                              for k in best[s]], index=best[a], columns=[a]) 
    #                             for a in best], axis=1).reindex(columns=sorted(best))
