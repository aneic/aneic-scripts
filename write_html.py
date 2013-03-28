'''
Generate html file of results
'''
import os
import sys
import pickle
import gzip
from string import Template
import numpy as np
import pandas as pd

sys.path.append('./aneic-core/src')
from aneic import mfm
from aneic import mutual

# Template string for HTML for page
doc_tpl = Template('''\
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
    <link rel="stylesheet" type="text/css" href="layout.css"/>
</head>
<body>
    <h2>$K Clusters</h2>
    <br>
    <table>
$headers
$rows
    </table>
</body>
</html>
''')

# Template for table rows
row_tpl = Template('''\
        <tr>
$cells
        </tr>\
''')

# Template for table cells (cluster descriptions)
desc_tpl = Template('''\
            <td>
$desc
            </td>\
''')

# Template for table cells (individual objects)
cell_tpl = Template('''\
            <td>
                <h3>$name</h3>
                <br>
                <a href="$url">
                    <img src="$url">
                </a>
$desc
            </td>\
''')

# Template for empty cells
cell_empty = '''\
            <td>
            </td>\
'''

def mutual_inf_dz(real, cat, theta, gamma, bins=10):
    mixz, miyz = mutual.MI0dz(real, cat, 
                    theta['mu'], theta['l'], theta['rho'], theta['pi'], 
                    gamma, bins=bins)
    return pd.concat([mixz, miyz])

def write_html(filename, real, cat, theta, gamma, mi=None, maxft=5, 
    base_url='%s.jpg'):
    if not mi is None:
        # get sorted list of top features
        top = mi.order(ascending=False).index
        # get feature counts per cluster
        real_count = pd.DataFrame((np.asarray(~pd.isnull(real))[:,None,:] 
                                   * np.asarray(gamma)[:,:,None]).sum(0),
                        index=gamma.columns, columns=real.columns)
        cat_count = pd.DataFrame((np.asarray(cat.fillna(0))[:,None,:] 
                                   * np.asarray(gamma)[:,:,None]).sum(0),
                        index=gamma.columns, columns=cat.columns)
    else:
        top = []

    # number of objects and clusters
    N = len(real)
    K = theta['mu'].shape[1]

    # build header strings
    headers = ['        <th>Cluster %d</th>' % k for k in range(K)]

    # build strings for description cells of each cluster
    if not mi is None:
        descs = []
        for k in range(K):
            indent16 = ' '.join(['']*16) 
            desc = '%s<p>Objects: %.1f</p>' \
                    % (indent16, gamma.ix[:,k].sum())
            f = 0
            for ft in top:
                info = ''
                if ft in real.columns:
                    info = '%.2f +/- %.2f (%d)' \
                           % (theta['mu'].ix[ft,k],
                              theta['l'].ix[ft,k]**-0.5,
                              real_count.ix[k,ft])
                if ft in cat.columns.levels[0]:
                    infos = ['%s: %.1f' % p for p in zip(cat[ft].columns, 
                                                         cat_count.ix[k,ft])]
                    info = ', '.join(infos)
                if info:
                    desc += '\n%s<p>%s: %s</p>' \
                            % (indent16, ft, info)
                    f +=1
                if f == maxft:
                    break
            descs.append(desc_tpl.substitute(desc=desc))                

    # build strings for objects cells of each cluster
    cells = [[] for k in range(K)]
    for k in range(K):
        # find objects in cluster
        msk = np.array(gamma.idxmax(axis=1) == k)
        g = gamma[msk]
        h = (- (g * np.log(g)).sum(axis=1))
        h.sort()
        for nm in h.index:
            url = base_url % nm
            # add gamma to info block
            indent16 = ' '.join(['']*16) 
            gstrings = ['%.2f' % g for g in gamma.ix[nm, :]]
            desc = '%s<p>weights: %s</p>' \
                    % (indent16, ', '.join(gstrings)) 
            # add info on top features if available
            f = 0
            for ft in top:
                val = ''
                if ft in real.columns and pd.notnull(real.ix[nm,ft]):
                    val = '%.2f' % real.ix[nm,ft]
                if ft in cat.columns.levels[0] and pd.notnull(cat.ix[nm,ft]).any():
                    val = '%s' % cat.ix[nm,ft][cat.ix[nm,ft].nonzero()[0]].index[0]
                if val:
                    desc += '\n%s<p>%s: %s (%.2f)</p>' \
                            % (indent16, ft, val, mi[ft])
                    f +=1
                if f == maxft:
                    break
            # add to cells
            cell = cell_tpl.substitute(name=nm, 
                                       url=url, 
                                       desc=desc)
            cells[k].append(cell)

    # build row strings
    R = max([len(c) for c in cells])
    rows = []
    for r in range(R):
        row = []
        for k in range(K):
            if r < len(cells[k]):
                row.append(cells[k][r])
            else:
                row.append(cell_empty)
        rows.append(row_tpl.substitute(cells='\n'.join(row)))
    if not mi is None:
        rows.insert(0, row_tpl.substitute(cells='\n'.join(descs)))

    # build page string
    doc = doc_tpl.substitute(K = str(K),
                             headers = '\n'.join(headers), 
                             rows = '\n'.join(rows)) 

    # write output
    f = open(filename, 'w')
    f.write(doc)
    f.close()

if __name__ == '__main__':
    # parse command-line args
    from optparse import OptionParser
    usage = "usage: %prog [options] files"
    parser = OptionParser(usage=usage)
    parser.add_option('-o', '--output-dir',
                      metavar="DIR", help="output directory", dest='outdir')
    parser.add_option('-i', '--image-dir',
                      metavar="DIR", help="image directory", dest='imdir')
    (opt, args) = parser.parse_args()

    # find files matching pattern for each supplied argument
    import glob
    files = []
    for pattern in args:
        files += glob.glob(pattern)

    # create output dir if necessary
    import os
    if opt.outdir and files:
        html_dir = opt.outdir
        if not os.path.isdir(html_dir):
            os.makedirs(html_dir)
    else:       
        html_dir = os.getcwd()
    
    im_dir = opt.imdir if opt.imdir else './aneic-core/data/images/'
    base_url = os.path.relpath(im_dir, html_dir) + '/%s.jpg'

    # write out html 
    data = None
    for fname in files:
        # load output
        mfm_output = pickle.load(gzip.open(fname, 'r'))
        # get dataset
        real = mfm_output['real']
        cat = mfm_output['cat']
        # get parameters
        theta = mfm_output['theta']
        # get state assignments
        gamma = mfm_output['gamma']
        # get mutual information
        midz = mutual_inf_dz(real, cat, theta, gamma, bins=5)
        # if data is None:
        #     # get path
        #     path = os.getcwd()
        #     # load dataset
        #     opts = mfm_output['opts']
        #     # load data
        #     data = pd.read_csv('%s/%s' %(path, opts['filename']), index_col=1, 
        #                        na_values=opts['na_values'])\
        #                .drop(opts['drop_feats'], axis=1)\
        #                .drop(opts['drop_objs'], axis=0)
        #     # split real and cat feats
        #     cat = mfm.convert_indicator(data[data.columns[data.dtypes == np.object]])
        #     real = data[data.columns[data.dtypes == np.float64]]
        # write html
        hname = fname.replace('.pk.gz', '.html').replace('results','html')
        base_path = os.path.relpath('./aneic-core/data/images/', html_dir)
        write_html(hname, real, cat, theta, gamma, 
            mi=midz, base_url=base_url)
