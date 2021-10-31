import numpy as np
import os
import glob
import cPickle as pickle

"""
this file serves for printing evaluation results
"""

def getOneMetric(perf):
    """
    prepare the format of one metric, e.g., MAP, precision-recall, etc.
    perf: one float value or array
    """
    ret=""
    if type(perf) is np.ndarray:
        #for precison-recall curve or recal-candidate curve
        if perf.ndim>1:
            for i in range(perf.ndim):
                for v in perf[i]:
                    ret+="%.4f " % v
                ret+="\t"
            #ret+="\n"
        else:
            for v in perf:
                ret+="%.4f " % v
            #ret+="\n"
    else:
        #for MAP 
        ret+="%.4f " % perf
    return ret


def printPerf(dir, ord,  name, id=None):
    files=os.listdir(dir)
    ret=[f for f in files if f.startswith(name)]
    if len(ret)==0:
        return
    ret.sort()
    ret.sort(key=len)
    if id is not None:
        ret=[ret[id]]

    if 'cross' in name:
    #print cross-modal search performance
        outImg="%-10s " % (name+'-qimg')
        outTxt="%-10s " % (name+'-qtxt')
        for f in ret:
            #evaluations are performed multiple times during training, there would be multiple performance files
            with open(os.path.join(dir, f), 'r') as fd:
                perf=pickle.load(fd)
            outImg+=getOneMetric(perf[0][ord])
            outTxt+=getOneMetric(perf[1][ord])
        print outImg
        print outTxt
    else:
    #print single-modal search performance
        outstr="%-10s " % name
        for f in ret:
            with open(os.path.join(dir, f), 'r') as fd:
                perf=pickle.load(fd)
            outstr+=getOneMetric(perf[ord])
        print outstr

      
def printPerfForAll(dir, ord, id=None):
    """
    print metrics onto screen
    dir: directory for performance results
    ord: metrics are stoed in one array, order is the pos of the metric to be printed, e.g., map is at 1st position
    """
    #prefix of performance files, to print metrics line by line
    mode=['iae', 'tae','isae', 'tsae','qimgV', 'qtxtV', 'crossV',  'qimgT', 'qtxtT','crossT'] #'rand'
    for m in mode:
        printPerf(dir, ord, m, id)
