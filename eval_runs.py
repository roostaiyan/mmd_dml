'''
Created on Feb 1, 2016

@author: ehsan
'''
import numpy as np
import scipy.io as sio
import os

def computeMap(query_label, target):
    result = (target == query_label)
    cumsum=result.cumsum(axis=1)
    prec=cumsum / np.arange(1.0,1+result.shape[1])
    total_relevant=cumsum[:,-1]+1e-5
    ap=np.sum((prec*result),axis=1)/total_relevant
    return np.mean(ap);

datasets = ['cal10','cal20','cal50','corel5k','indoor']
split_cnt = 5
try:
	#datasets = ['corel5k']
	config_cnt = 1
	split_dirs = ['msaept', 'msaeptsp2', 'msaeptsp3', 'msaeptsp4', 'msaeptsp5']
	mode = 'test'
	print mode
	for ds in datasets:
	    print ds
	    for c in xrange(config_cnt):
    		temp_map = 0.0
    		for sp in xrange(split_cnt):
    			#base_dir = os.path.join('..', split_dirs[sp], 'data', ds, 'output', 'output'+str(c),'msae','perf', mode)
    			base_dir = os.path.join('targets', 'main_runs', 'split'+str(sp+1), ds, 'output'+str(c),'msae','perf', mode)
    			#base_dir = os.path.join('targets', 'number_of_layers', ds, 'output'+str(c),'msae','perf', mode)
    			#base_dir = os.path.join('targets', 'final_dim', ds, 'output'+str(c),'msae','perf', mode)
    			query_label = np.load(os.path.join(base_dir, 'querylabel.npy'))[:,np.newaxis]
    			target = np.load(os.path.join(base_dir, 'target.npy'))
    			if mode=='val':
    			    target = target[:,1:]#omit the first col for val
    		#	print query_label.shape
    		#	print target.shape
                	print computeMap(query_label,target)
                temp_map += computeMap(query_label,target)/split_cnt
    		#print temp_map


except Exception,e: print str(e)

print ''
print 'OMKS-50'
for ds in datasets:
	print ds
	tmp=0.0
	for i in range(split_cnt):
		base_dir = '/home/mll/Desktop/Imani/OMKS/targets/50/'+ds+'/'+str(i+1)+'/0.5/'
		query_label = sio.loadmat(base_dir+'query_label.mat')['l1']
		target = sio.loadmat(base_dir+'target.mat')['sorted']
		tmp += computeMap(query_label,target)/split_cnt
	print tmp
