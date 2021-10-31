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

import gnumpy as gp
import numpy as np
import sys
from model import Model
from datahandler import DataHandler
import pickle

class SAE(Model):
    """
    stacked autoencoder
    """
    def __init__(self, config, name, prefix="ae"):
        self.prefix = prefix
        self.config = config
        self.name = name
        super(SAE, self).__init__(config, name)
        #depth is the #layers=#autoencoders+1
        self.depth=int(self.readField(config, name, "depth"))
        self.ae=[None]
        reset=self.readField(config, name ,"reset_hyperparam")

        #Ehsan
        #load jdepth
        if prefix == "jae":
            self.depth=int(self.readField(config, name, "initial_joint_depth"))    
        #load each autoencoder
        for i in range(1,self.depth):
            ae=self.loadModel(config, self.readField(config, name, prefix+str(i)))
            self.sections.append(ae.name)
            if reset!="False":
                ae.resetHyperParam(config, reset)
            self.ae.append(ae)
        self.vDim=self.ae[1].vDim
        self.hDim=self.ae[-1].hDim

    def addAE(self, pretrain='ae'):
        ae=self.loadModel(self.config, self.readField(self.config, self.name, self.prefix+str(self.depth)))
        print 'pretrain is:::::', pretrain
        if pretrain=='ae':
            trainDataSize=int(self.readField(self.ae[-1].config, self.ae[-1].name, "train_size"))
            numBatch = trainDataSize / self.ae[-1].batchsize
            self.ae[-1].extractTrainReps(self.ae[-1].trainDataLoader, numBatch)
            ae.initWithPCA()
            ae.train()
        elif pretrain=='mse':
            print 'mse!!'
            ae.initWithMSE()
        else:
            ae.initRandom()
        self.depth += 1
        self.sections.append(ae.name)
        self.ae.append(ae)
        self.hDim=self.ae[-1].hDim
        
    def forward(self, dat, training=False):
        """forward through latent and reconstruction layers"""
        a=self.forward2Top(dat,training)
        a=self.backward2Bottom(a)
        return a

    def forward2Top(self, dat, training=False):
        """compute latent layers"""
        depth = self.depth
        a=[]
        if training:
            a.append(self.ae[1].getCorruptedInput(dat))
        else:
            a.append(dat)
        for i in range(1, depth):
            a.append(self.ae[i].forwardOneStep(a[-1]))
        a[0]=dat
        return a

    def backward2Bottom(self, a):
        """compute reconstruction layers"""
        for i in range(self.depth-1, 0, -1):
            a.append(self.ae[i].backwardOneStep(a[-1]))
        return a

    def getCost(self, param, a0,factor=1.0):
        """objective loss=reconstruct error+L2"""
        self.splitParam(param)
        a=self.forward(a0)
        cost=self.ae[1].getErrorLoss(a[0],a[-1],factor)
        for i in range(1,self.depth):
            cost+=self.ae[i].getWeightLoss(self.ae[i].W1,self.ae[i].W2)
        return cost,a

    def splitParam(self, param):
        """
        split parameters in array into [W1,b1],[W2,b2]
        """
        k=0
        for i in range(1,self.depth):
            s=self.ae[i].W1.size
            self.ae[i].W1=param[k:k+s].reshape(self.ae[i].W1.shape)
            k=k+s
            s=self.ae[i].b1.size
            self.ae[i].b1=param[k:k+s].reshape(self.ae[i].b1.shape)
            k=k+s
        if len(param)>k: 
            for i in range(self.depth-1,0,-1):
                s=self.ae[i].W2.size
                self.ae[i].W2=param[k:k+s].reshape(self.ae[i].W2.shape)
                k=k+s
                s=self.ae[i].b2.size
                self.ae[i].b2=param[k:k+s].reshape(self.ae[i].b2.shape)
                k=k+s

    def combineParam(self,down=True):
        param=[]
        for i in range(1, self.depth):
            param.append(self.ae[i].W1)
            param.append(self.ae[i].b1)
        if down:
            for i in range(self.depth-1,0,-1):
                param.append(self.ae[i].W2)
                param.append(self.ae[i].b2)
        return self.vectorParam(param)

    def computeNumericGradient(self, input, factor=1.0, eps=1e-4, sampleNum=500):
        """
        compute gradients throught numeric way for gradient check
        gradient of J w.r.t. x computed by (J(x+eps)-J(x-eps))/2eps
        only check param at sampleNum positions
        J=0.5*(a[0]-a[-1])**2+WeightCost
        """
        param=self.combineParam()
        plen=param.size
        if factor==0:
            plen=plen/2
        sample=np.random.randint(0,plen,sampleNum)
        grad=gp.zeros(sampleNum)
        for (i,idx) in enumerate(sample):
            if i%100==0:
                sys.stdout.write('.')
                sys.stdout.flush()
            q=gp.zeros(param.shape)
            q[idx]=eps
            p1=param+q
            p2=param-q
            c1,_=self.getCost(p1, input,factor)
            c2,_=self.getCost(p2, input,factor)
            grad[i]=(c1-c2)/(2.0*eps)
        print "end"
        return grad, sample

    def gradientCheck(self, dat):
        """
        check gradient by comparing with numeric computing
        it should be done on cpu
        """
        print "doing gradient check..."

        a=self.forward(dat)
        pgrad=self.computeGrads(a)
        pgrad=self.vectorParam(pgrad)

        pnumeric,idx=self.computeNumericGradient(dat)
        pgrad=pgrad[idx]
        diff= (pgrad-pnumeric).euclid_norm()/(pgrad+pnumeric).euclid_norm()
        print "the diff is %.15f, which should be very small" % diff

    def computeGrads(self, a, diffgrad=None,factor=1.0):
        """
        diffgrad is dDiff/da', Diff is diff of top layer, a' is top layers
        for single sae diffgrad=0; for msae diffgrad is generated by 0.5*||img-txt||^2
        return w1,b1...w2,b2... from bottom to top to bottom
        """
        assert(diffgrad!=None or  factor>0)
        aes=self.ae
        grad=[]
        d=[0]*self.depth
        topidx=self.depth-1

        if factor>0:
            #compute derivatives of reconstruction layers from L_r
            d[0]=aes[1].computeDlast(a[0],a[-1],factor)
            for i in range(1,self.depth):
                grad.append(aes[i].getbGradient(d[i-1]))
                grad.append(aes[i].getWGradient(d[i-1],a[-1-i],aes[i].W2))
                if i+1<self.depth:
                    d[i]=aes[i+1].computeD(a[-1-i],d[i-1],aes[i].W2)
            d[topidx]=gp.dot(d[topidx-1],aes[topidx].W2.T)

        if diffgrad is not None:
            #combine derivates from L_d
            d[topidx]+=diffgrad
        d[topidx]*=aes[topidx].getActivationGradient(a[topidx])

        for i in range(self.depth-1,0,-1):
            #compute derivates of latent layers
            grad.append(aes[i].getbGradient(d[i]))
            grad.append(aes[i].getWGradient(d[i],a[i-1],aes[i].W1))
            if i>1:
                d[i-1]=aes[i-1].computeD(a[i-1],d[i],aes[i].W1)
        grad.reverse()
        return grad

    def updateParams(self, epoch, g, aes, backprop=False):
        """
        g: from bottom to top, w1,b1..; from top to bottom, w2,b2...
        """
        depth=self.depth
        
        if epoch < 0:#31:#not backprop:
            i = depth-1
            aes[i].updateParam(epoch,aes[i].W1,aes[i].incW1,g[2*i-2])
            aes[i].updateParam(epoch,aes[i].b1,aes[i].incb1,g[2*i-1])
            if len(g)>2*depth-2:
                g=g[depth*2-2:]
                k=depth
                aes[i].updateParam(epoch,aes[i].W2,aes[i].incW2,g[2*(k-i)-2])
                aes[i].updateParam(epoch,aes[i].b2,aes[i].incb2,g[2*(k-i)-1])
        else:            
            for i in range (1,depth):
                aes[i].updateParam(epoch,aes[i].W1,aes[i].incW1,g[2*i-2])
                aes[i].updateParam(epoch,aes[i].b1,aes[i].incb1,g[2*i-1])
            if len(g)>2*depth-2:
                g=g[depth*2-2:]
                k=depth
                for i in range(depth-1,0,-1):
                    aes[i].updateParam(epoch,aes[i].W2,aes[i].incW2,g[2*(k-i)-2])
                    aes[i].updateParam(epoch,aes[i].b2,aes[i].incb2,g[2*(k-i)-1])

    def trainOneBatch(self, input, epoch, computeStat=True):
        assert (self.batchsize == input.shape[0])
        dat = gp.as_garray(input)

        if self.debug:
            self.gradientCheck(dat)
            sys.exit(0)

        a=self.forward(input, training=True)
        grads=self.computeGrads(a)
        assert(len(grads)==4*self.depth-4)
        self.updateParams(epoch,grads,self.ae)
        
        if computeStat:
            perf=[self.ae[1].getErrorLoss(a[0], a[-1])]
            for i in range(1, self.depth):
                perf.append(self.ae[i].computeSparsity(a[i]))
            return np.array(perf)

    def getReps(self, dat):
        a=self.forward2Top(dat)
        return a[-1].as_numpy_array()

    def getDisplayFields(self):
        s="neighbor dist, epoch , rec err, sparsities"
        return self.depth,self.depth,s

    def doCheckpoint(self, outdir):
        """
        save all autoencoders along the path under the directory of sae
        """
        import os
        for i in range(1, self.depth):
            path=os.path.join(outdir,self.ae[i].name)
            self.ae[i].save(path)
        super(SAE,self).doCheckpoint(outdir)
