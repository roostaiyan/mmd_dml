"""
Copyright (C) 2014 Wei Wang (wangwei@comp.nus.edu.sg)

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
import numpy as np
import scipy.spatial
import sys
import os
import cPickle as pickle
#import time


class Evaluator(object):
    '''
    Evaluation metrics include  Mean Average Precision, precision-recall and recall-candidate ratio
    MAP=\sum_i^n ap(q_i)/n
    ap(q)=\sum_k p(k)*rel(k)/total_relevant
    p(k)=\sum_{j=1}^k rel(j)
    rel(k)=0 if not relevant, 1 if relevant
    the ground truth is calculated based on the label of data.
    if the query and the result share at least one concept, then
    the result is considered to be relevant
    '''

    def __init__(self,query_path,label,output_dir,name="eval",query_size=100, verbose=True):
        '''
        query_path: create and save query index(row No.) if query_path not exist;
                    all exps would use the same query index
        label_path: each row represents the concepts of one object, val:0/1
        output_dir: e.g., output/msae/perf/
        verbose: print intermediate info
        name: model name, e.g., msae
        '''

        self.name=name
        self.verbose=verbose
        if os.path.exists(query_path):
            qindex=np.load(query_path)
            self.query_size=len(qindex)
            #debug 
#             print self.query_size
#             print query_size
            assert(self.query_size==query_size)
        else:
            self.query_size=query_size
            #duplicate queries may exist
            qindex=np.random.randint(0,label.shape[0],self.query_size*2)
            #filter queries without labels
            nnz=np.where(np.sum(label[qindex],axis=1)>0)[0]
            qindex=qindex[nnz]
            #remove duplicates
            qindex=list(set(qindex))
            qindex=qindex[0:query_size]
            np.save(query_path, qindex)

        self.qindex=qindex
        self.label = label
        #two instances are relevant if they share at least one concept/label
        self.ground_th=np.dot(label[qindex],label.T)>0
        self.output_dir=output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        #evaluate random search
        self.evalRandom(0)
        
        #Ehsan
        self.mapArray = np.empty((0), np.float)
    
    def setTrainLabel(self, label2):
        self.trainLabel = label2
        self.ground_th=np.dot(self.label,label2.T)>0
        
#         print self.ground_th
#         print self.ground_th.shape
        
    def evalSingleModal(self, dat,step,name,metric='euclidean'):
        """
        single model search
        name:qimg,qtxt; or name of autoencoder
        """
        sorted_ret=search(dat[self.qindex],dat,metric=metric, verbose=self.verbose)
        perf=computePerformance(sorted_ret,self.ground_th)
        print ""
        print "MAP is: ", perf[0]
        self.mapArray = np.append(self.mapArray, perf[0])
#         with open(os.path.join(self.output_dir,name+str(step)),'w') as fd:
#             pickle.dump(perf, fd)
    
    #Ehsan
    def evalSingleModal2(self, qdat, traindat,step,name,metric='euclidean'):
        """
        single modal search
        name:qimg,qtxt; or name of autoencoder
        """
        sorted_ret=search(qdat,traindat,metric=metric, verbose=self.verbose)
        
        perf=computePerformance(sorted_ret,self.ground_th)
        print ""
        print "MAP is: ", perf[0]
        self.mapArray = np.append(self.mapArray, perf[0])
#         with open(os.path.join(self.output_dir,name+str(step)),'w') as fd:
#             pickle.dump(perf, fd)
            
    def evalClassification(self, data, label, epoch, name, metric='euclidean'):
        out = 1*(data==np.amax(data,axis=1, keepdims=True))
        print 'num: ', label.shape[0]
        correct = 0
        for i in xrange(label.shape[0]):
            correct += 1*((out[i,:]==label[i,:]).all())
        print "Classification acc is: ",(correct*1.0)/label.shape[0], ', ', correct
    
    #Ehsan
    def saveMaps(self, name):
        savepath = os.path.join(self.output_dir, name)
        np.save(savepath, self.mapArray)
        
    #Ehsan
    def saveTarget(self, qdat, traindat,metric='euclidean'):
        sorted_ret=search(qdat,traindat,metric=metric, verbose=self.verbose)
        sorted_result=sorted_ret.astype(int)
        sorted_target=np.zeros(sorted_result.shape)
        for i in range(sorted_result.shape[0]):
                sorted_target[i,:] = np.argmax(self.trainLabel[sorted_result[i,:]],axis=1)
#                 sorted_target[i]=self.ground_th[i][sorted_result[i]]
        np.save(os.path.join(self.output_dir,'target'),sorted_target)
        np.save(os.path.join(self.output_dir,'querylabel'),np.argmax(self.label,axis=1))
        print 'target saved in ', self.output_dir

    def evalCrossModal(self, images, text, step, suffix, metric='euclidean'):
        """
        cross modal search
        suffix :indicates validation or test
        step :the epoch id of training
        images: latent image features
        text: latent text features

        there are 4 searches:
        qimg->img, perf saved in qimg[Suffix][step] file
        qtxt->txt, perf saved in qtxt[Suffix][step] file
        qimg->txt, perf saved in cross[suffix][step] file
        qtxt->img, perf saved in cross[suffix][step] file
        """
        self.evalSingleModal(images, str(step), 'qimg'+suffix,metric=metric)
        self.evalSingleModal(text,str(step),'qtxt'+suffix,metric=metric)

        sorted_ret=search(images[self.qindex],text,metric=metric, verbose=self.verbose)
        imageTextPerf=computePerformance(sorted_ret,self.ground_th)

        sorted_ret=search(text[self.qindex],images,metric=metric, verbose=self.verbose)
        textImagePerf=computePerformance(sorted_ret,self.ground_th)

        with open(os.path.join(self.output_dir,'cross'+suffix+str(step)),'w') as fd:
            pickle.dump([imageTextPerf, textImagePerf], fd)
            
    

    def evalRandom(self, step):
        query=np.arange(self.ground_th.shape[1])
        rand_result=np.empty((len(self.qindex),len(query)))
        for i in range(len(self.qindex)):
            np.random.shuffle(query)
            rand_result[i]=query
        randPerf=computePerformance(rand_result, self.ground_th)
        with open(os.path.join(self.output_dir,'rand'+str(step)),'w') as fd:
            pickle.dump(randPerf, fd)

def search(query, data, metric='euclidean', verbose=True):
    """
    do search, return ranked list according to distance
    metric: hamming/euclidean
    query: one query per row
    dat: one data point per row
    """
    #calc dist of query and each data point
    if metric not in ['euclidean', 'hamming']:
        print 'metric must be one of (euclidean, hamming)'
        sys.exit(0)
    
    #b=time.clock()
    nquery = query/ np.linalg.norm(query,axis=1)[:,np.newaxis]
    ndata = data/ np.linalg.norm(data,axis=1)[:,np.newaxis]    
    dist=scipy.spatial.distance.cdist(nquery,ndata,metric)
    sorted_idx=np.argsort(dist,axis=1)
    #e=time.clock()

    if verbose:
        #calc avg dist to nearest 200  neighbors
        nearpoints=sorted_idx[:,0:200]
        d=[np.mean(dist[i][nearpoints[i]]) for i in range(nearpoints.shape[0])]
        sys.stdout.write('%.4f, '% np.mean(d))
        #print 'search time %.4f' % (e-b)
    return sorted_idx


def computePrecAndRecall(result):
    """
    compute precision and recall for all position
    """
    cumsum=result.cumsum(axis=1)
    prec=cumsum / np.arange(1.0,1+result.shape[1])
    total_relevant=cumsum[:,-1]+1e-5
    recall=cumsum / total_relevant[:,np.newaxis]
    return prec,recall
 

def computeMAP(result, k, prec=None):
    """
    compute map, if k==0, map for all results
    if k>0, map for top-k results
    return avgmap, avg prec@50  and map std
    """
    if k>0:
        result=result[:,0:k]
        prec=None
    total_relevant=np.sum(result, axis=1)+1e-5
    if prec is None:
        prec,_=computePrecAndRecall(result)
    ap=np.sum((prec*result),axis=1)/total_relevant
    return np.average(ap),np.average(prec[:,50]),np.std(ap)
#     return np.average(ap),np.average(ap),np.std(ap)


def computePrecRecall(result, prec=None, recall=None):
    """
    compute avg precison recall at 0,0.1,0.2,..1.0
    to plot prec-recall curve
    """
    if prec is None:
        prec,recall=computePrecAndRecall(result)
    recall10=np.zeros((recall.shape[0],11))
    prec10=np.zeros(recall10.shape)
    recthreshold=np.arange(0.0,1.1,0.1)-1e-5
    for i in range(recall.shape[0]):
        if recall[i][-1]>0:
            recindex=np.searchsorted(recall[i],recthreshold)
            recall10[i]=recall[i][recindex]
            prec10[i]=prec[i][recindex]
    recallAvg=np.average(recall10,0)
    precAvg=np.average(prec10,0)
    return np.array([precAvg, recallAvg])


def computeRecallRatio(result, recall=None):
    """
    compute recall @ pos [0.2,0.4,0.6,0.8,1.0] * result size
    return avg recall, which would be used to plot recall-checked_data_ratio fig
    """
    if recall is None:
        prec,recall=computePrecAndRecall(result)
    pos=(recall.shape[1]-1)*np.array([0.2,0.4,0.6,0.8,1.0])
    pos=pos.astype(int)
    recall=recall[:,pos]
    return np.average(recall, axis=0)


def computePerformance(sorted_result, ground_th):
    '''
    sorted_result: each row is the sorted data point index for one query
    ground_th: each row is the ground truth for one query, 1:relevant, 0:irrelevant
    '''
    sorted_result=sorted_result.astype(int)
    sorted_target=np.zeros(sorted_result.shape)
    for i in range(sorted_result.shape[0]):
            sorted_target[i]=ground_th[i][sorted_result[i]]
    prec,recall=computePrecAndRecall(sorted_target)
    map,prec50,mapstd=computeMAP(sorted_target, k=0, prec=prec)
    precrecall=computePrecRecall(sorted_target,prec=prec,recall=recall)
    recallratio=computeRecallRatio(sorted_target,recall)

    return [map,recallratio,precrecall,prec50,mapstd]

def searchPerf(query,data,gnd,name=''):
    '''
    search query against data, output perf directly
    '''
    sorted_ret=search(query,data)
    perf=computePerformance(sorted_ret,gnd)
    print perf
