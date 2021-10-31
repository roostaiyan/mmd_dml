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
import scipy.io
import configparser
import argparse
import os
import time

import evaluate
import util


def train(configPath, name):
    useGpu = os.environ.get('GNUMPY_USE_GPU', 'auto')
    if useGpu=="no":
        mode="cpu"
    else:
        mode="gpu"

    print '========================================================'
    print 'train %s' % name
    print "the program is on %s" % mode
    print '======================================================='

    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(configPath)

    model_name=config.get(name, 'model')
    if model_name == "ae":
        from ae import AE
        model = AE(config, name)
    elif model_name == "lae":
        from lae import LAE
        model = LAE(config, name)
    elif model_name == "pcaae":
        from pcaae import PCAAE
        model = PCAAE(config, name)
    elif model_name == "flae":
        from flae import FLAE
        model = FLAE(config, name)
    elif model_name == "pae":
        from pae import PAE
        model = PAE(config, name)
    elif model_name== "sae":
        from sae import SAE
        model=SAE(config, name)
    elif model_name== "msae":
        from msae import MSAE
        model=MSAE(config, name)
 
    model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="driver program for training")
    parser.add_argument("-a", nargs='+', help="autoencoder names(sections in config file)+config file path")
    parser.add_argument("-e", nargs='+', help="extract reps of input(training/test) data: input model file, img path, txt path, output dir")
    parser.add_argument("-p", nargs='+', help="print performance results:[dir, metric(map/precrecall/recall)]")
    parser.add_argument("-s", nargs='+', help="search/evaluation against latent features of test dataset. [query path;ground truth;latent features;metric; query size]; \
                        latent features are either from matlab file with Bx_te=image features and By_te=text features, or in separate .npy file; perf results are saved in ./tmp")
    args = parser.parse_args()

    #train autoencoders, either sae or msae
    if args.a:
        for i in range (len(args.a)-1):
            start=time.time()
            #the last arg is config file path
            train(args.a[-1],args.a[i])
            end=time.time()
            #print 'elapsed time for %s is %f min' % (args.a[i],(end-start)/60.0)

    #extract latent features, i.e., top layer latent representation
    if args.e:
        from model import Model
        ae=Model.load(args.e[0])
        if len(args.e)>1:
            #for test dataset
            normalizeImg=ae.str2bool(ae.readField(ae.config, ae.name, "normalize"))
            if normalizeImg:
                #args.e[3] is path for stat file
                assert(len(args.e)==5)
                imgcode, txtcode=ae.inference(args.e[1],args.e[2],args.e[3])
            else:
                imgcode, txtcode=ae.inference(args.e[1],args.e[2])
            outdir=args.e[-1]
            np.save(os.path.join(outdir,"img"),imgcode)
            np.save(os.path.join(outdir,"txt"),txtcode)
            
   
    #print performance
    if args.p:
        print '\n##Performance Results##'
        if args.p[1]=="map":
            util.printPerfForAll(args.p[0],0)
        elif args.p[1]=="recall":
            util.printPerfForAll(args.p[0], 1)
        elif args.p[1]=="precrecall":
            util.printPerfForAll(args.p[0], 2)
        else:
            print "wrong metric, should be 'map' or 'precrecall' or 'recall'"

    #search with real-valued/binary latent features
    if args.s:
        qpath=args.s[0]
        label=np.load(args.s[1])
        metric=args.s[-2] #'hamming' or 'euclidean'
        qsize=int(args.s[-1])
        #query file will be created if not exists
        searcher=evaluate.Evaluator(qpath,label,"tmp",query_size=qsize)

        if args.s[2].endswith(".mat"):
            dat=scipy.io.loadmat(args.s[2])
            img=dat['Bx_te']
            txt=dat['By_te']
        else:
            img=np.load(args.s[2])
            txt=np.load(args.s[3])
            
        assert(img.shape==txt.shape)
        #transpose if necessary 
        if img.shape[0]<img.shape[1]:
            img=img.T
            txt=txt.T
        searcher.evalCrossModal(img,txt,'','T', metric=metric)
