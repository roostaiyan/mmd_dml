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
    
from model import Model
import gnumpy as gp
import numpy as np
from sae import SAE
import evaluate
import sys
import os
from datahandler import DataHandler
import matplotlib.pyplot as plt
from itertools import product
from operator import add
from numpy import diff
from gnumpy import garray

class MSAE(Model):
    """
    multi-modal stacked autoencoder
    """

    def __init__(self, config, name="msae"):
        super(MSAE,self).__init__(config, name)

        self.depth=int(self.readField(config, name, "depth"))
        #joint depth
        self.ijdepth=int(self.readField(config, name, "initial_joint_depth"))
        self.max_jdepth=int(self.readField(config, name, "joint_depth"))
        self.has_joint = self.str2bool(self.readField(self.config, self.name, "has_joint"))

        self.names = self.readField(config, name, "modal_names").split(',')
        self.prefixes = [x+'ae' for x in self.names]
        self.saeNames = [x+'sae' for x in self.names]
        self.modalsCnt = len(self.names) 
        print self.names
        print self.prefixes
        print self.saeNames
        print self.modalsCnt
        
        self.initsaes() 

        dimsstr = (self.readField(self.config,self.jsae.ae[1].name,'train_dims')).split(',')
        self.dims = [int(i) for i in dimsstr]
        self.dims.insert(0,0)
        self.dims = np.cumsum(self.dims)
        
        self.epoch=0
        self.statesIdx=0
        states=self.readField(config, name, 'states')
        fields=states.split(',')
        self.states=[] 
        assert(len(fields)%7==0)

        #states indicate which sae to fix, which to adjust
        for i in range(len(fields)/7):
            k=0
            state=[]
            state.append(fields[i*7+k])
            state.append(self.str2bool(fields[i*7+k+1]))
            state.append(self.str2bool(fields[i*7+k+2]))
            state.append(float(fields[i*7+k+3]))
            state.append(float(fields[i*7+k+4]))
            state.append(float(fields[i*7+k+5]))
            state.append(int(fields[i*7+k+6]))
            self.states.append(state)
        #print self.states

        sim_diff_coststr = self.readField(config, name, 'sim_diff_cost').split(',')
        sim_diff_cost = []
        for i in range(len(sim_diff_coststr)):
            sim_diff_cost.append(float(sim_diff_coststr[i]))
            
        dis_diff_coststr = self.readField(config, name, 'dis_diff_cost').split(',')
        dis_diff_cost = []
        for i in range(len(dis_diff_coststr)):
            dis_diff_cost.append(float(dis_diff_coststr[i]))
            
        rec_coststr = self.readField(config, name, 'rec_cost').split(',')
        rec_cost = []
        for i in range(len(rec_coststr)):
            rec_cost.append(float(rec_coststr[i]))
            

        self.params = list(product(sim_diff_cost,dis_diff_cost,rec_cost))
        self.paramInd = 0
        
        for s in self.saes:
            self.sections.extend(s.sections)
        self.sections.extend(self.jsae.sections)

    def initsaes(self):
        self.saes = [[] for x in xrange(self.modalsCnt)]
        for i in range(self.modalsCnt):
            self.saes[i] = self.createsae(self.prefixes[i], self.saeNames[i])
        self.jsae = self.createsae("jae", "jsae")
            
    def createsae(self, prefix, saeName):
        if self.config.has_option(self.name, saeName):
            saepath=self.readField(self.config, self.name, saeName)
            sae=self.loadModel(self.config, saepath)
            reset=self.readField(self.config, self.name, "reset_hyperparam")
            if reset!="False":
                for ae in sae.ae[1:]:
                    ae.resetHyperParam(self.config, reset)
            return sae 
        else:
            return SAE(self.config, self.name, prefix=prefix)
    
    #Ehsan
    def computeGrads2(self, a, ja, diffgrad, recf):
#         assert(recf==0)
        aes = []
        grad = []
        d = []
        for m in xrange(self.modalsCnt):
            aes.append(self.saes[m].ae)
            grad.append([])
            d.append([0]*self.depth)
        jaes=self.jsae.ae
        jgrad=[]
        jd=[0]*self.jdepth
        topidx=self.depth-1
        topjidx=self.jdepth-1
        
        if recf>0:
            for m in xrange(self.modalsCnt):
                #compute derivatives of reconstruction layers from L_r for saes
                d[m][0]=aes[m][1].computeDlast(a[m][0],a[m][-1],recf)
                for i in range(1,self.depth):
                    grad[m].append(aes[m][i].getbGradient(d[m][i-1]))
                    grad[m].append(aes[m][i].getWGradient(d[m][i-1],a[m][-1-i],aes[m][i].W2))
                    if i+1<self.depth:
                        d[m][i]=aes[m][i+1].computeD(a[m][-1-i],d[m][i-1],aes[m][i].W2)
                d[m][topidx]=aes[m][topidx].computeD(a[m][-1-topidx],d[m][topidx-1],aes[m][topidx].W2)
#                 d[m][topidx]=gp.dot(d[m][topidx-1],aes[m][topidx].W2.T)
            
            #compute derivatives of reconstruction layers from L_r for jsae
            if self.has_joint:
                jd[0] = gp.concatenate(tuple(e[self.depth-1] for e in d),axis=1)
    
    #             jd[0] = jaes[1].computeDlast(ja[0],ja[-1],recf)
                
                for i in range(1,self.jdepth):
                    jgrad.append(jaes[i].getbGradient(jd[i-1]))
                    jgrad.append(jaes[i].getWGradient(jd[i-1],ja[-1-i],jaes[i].W2))
                    if i+1<self.jdepth:
                        jd[i]=jaes[i+1].computeD(ja[-1-i],jd[i-1],jaes[i].W2)
                jd[topjidx]=gp.dot(jd[topjidx-1],jaes[topjidx].W2.T)
                
        
        #add diffgrad to generative loss
        if self.has_joint:
            if diffgrad is not None:
                    jd[topjidx]+=(diffgrad+self.sparsityFactor*(2*ja[topjidx]/(1+ja[topjidx]*ja[topjidx])))*jaes[topjidx].getActivationGradient(ja[topjidx])
            #backprop in jsae
            for i in range(self.jdepth-1,0,-1):
                #compute derivates of latent layers
                jgrad.append(jaes[i].getbGradient(jd[i]))
                jgrad.append(jaes[i].getWGradient(jd[i],ja[i-1],jaes[i].W1))
                if i>1:
                    jd[i-1]=jaes[i-1].computeD(ja[i-1],jd[i],jaes[i].W1)+self.sparsityFactor*(2*ja[i-1]/(1+ja[i-1]*ja[i-1]))*jaes[topjidx].getActivationGradient(ja[i-1])
                
        #propagate to isae and tsae
        if diffgrad is not None:
            if self.has_joint:
                transD = aes[0][topidx].computeD((ja[0]),jd[1],(jaes[1].W1))
            else:
                transD = (diffgrad)*aes[0][topidx].getActivationGradient(gp.concatenate(tuple(e[topidx] for e in a),axis=1)) # no sparsity
        for m in xrange(self.modalsCnt):
            if diffgrad is not None:
            #combine derivates from L_d
                d[m][topidx]+= transD[:,self.dims[m]:self.dims[m+1]]
            d[m][topidx]*=aes[m][topidx].getActivationGradient(a[m][topidx])

            for i in range(self.depth-1,0,-1):
                #compute derivates of latent layers
                grad[m].append(aes[m][i].getbGradient(d[m][i]))
                grad[m].append(aes[m][i].getWGradient(d[m][i],a[m][i-1],aes[m][i].W1))
                if i>1:
                    d[m][i-1]=aes[m][i-1].computeD(a[m][i-1],d[m][i],aes[m][i].W1)

        for m in xrange(self.modalsCnt):
            grad[m].reverse()
        jgrad.reverse()
        return grad,jgrad
        
    
    #Ehsan
    def getSinglePathGrad2(self, a, ja, sim, other, recf, sim_diff_factor, dis_diff_factor):
        """
        ia:image ae data
        ta:text ae data
        ja:joint ae data
        sim: should this be similar to other
        other:output of jae given other element of the pair
        """
        recloss = []
        for m in xrange(self.modalsCnt):
            recloss.append(0)
        if recf>0:
            for m in xrange(self.modalsCnt):    
#                 a[m]=self.saes[m].backward2Bottom(a[m])
                recloss[m]=self.saes[m].ae[1].getErrorLoss(a[m][0],a[m][-1],recf)
            
#             ja=self.jsae.backward2Bottom(ja)
#             jrecloss=self.jsae.ae[1].getErrorLoss(ja[0],ja[-1],recf)
        if sim_diff_factor==0 and dis_diff_factor==0:
            diffgrad=None
        else:
            if(sim):
 
                if self.has_joint:
                    npj = ja[self.jdepth-1]#.as_numpy_array()
                else:
                    npj = gp.concatenate(tuple(e[self.depth-1] for e in a),axis=1)#.as_numpy_array()
                npo = other#.as_numpy_array()
                jsum = ((npj**2).sum(axis=1))**0.5#(np.linalg.norm(npj,axis=1))
                nj = (npj/ jsum[:,gp.newaxis])
                osum = ((npj**2).sum(axis=1))**0.5#(np.linalg.norm(npo,axis=1))
                no = (npo/ osum[:,gp.newaxis])
#                 jsum = gp.as_garray(jsum)
#                 osum = gp.as_garray(osum)
#                 nj = gp.as_garray(nj)
#                 no = gp.as_garray(no)
                   
                tmp = gp.sum(nj*no, axis=1)
                tmp = tmp.reshape(tmp.shape + (1,))
                tmp = gp.garray(tmp)
                tmp = (tmp*nj-no)
                tmp = tmp / jsum[:,gp.newaxis]
                   
                dist = (1-gp.sum(nj*no,axis=1))
                dist = dist>0.034
                diffgrad = gp.zeros(nj.shape)
                for i in xrange(self.batchsize):
                    if dist[i]:
                        diffgrad[i,:] = (tmp[i,:])
                        
                diffgrad=sim_diff_factor*diffgrad/self.batchsize
                
            else:
                
                if self.has_joint:
                    npj = ja[self.jdepth-1]#.as_numpy_array()
                else:
                    npj = gp.concatenate(tuple(e[self.depth-1] for e in a),axis=1)#.as_numpy_array()
                npo = other#.as_numpy_array()
                jsum = ((npj**2).sum(axis=1))**0.5#(np.linalg.norm(npj,axis=1))
                nj = (npj/ jsum[:,gp.newaxis])
                osum = ((npj**2).sum(axis=1))**0.5#(np.linalg.norm(npo,axis=1))
                no = (npo/ osum[:,gp.newaxis])
#                 jsum = gp.as_garray(jsum)
#                 osum = gp.as_garray(osum)
#                 nj = gp.as_garray(nj)
#                 no = gp.as_garray(no)
   
                tmp = gp.sum(nj*no, axis=1)
                tmp = tmp.reshape(tmp.shape + (1,))
                tmp = (tmp*nj-no)
                tmp = -1*tmp / jsum[:,gp.newaxis]                 
                   
                dist = (1-gp.sum(nj*no,axis=1))
                dist = dist<0.1
                diffgrad = gp.zeros(nj.shape)
                for i in xrange(self.batchsize):
                    if dist[i]:
                        diffgrad[i,:] = (tmp[i,:])
                
                diffgrad=dis_diff_factor*diffgrad/self.batchsize
                
        g, jg =self.computeGrads2(a,ja,diffgrad, recf)
        return g, jg, recloss
    
        #Ehsan
    def getClassificationGrad(self, a, ja, label, recf=0.0, diff_factor=1.0):
        """
        """
        recloss = []
        for m in xrange(self.modalsCnt):
            recloss.append(0)
        if recf>0:
            for m in xrange(self.modalsCnt):    
#                 a[m]=self.saes[m].backward2Bottom(a[m])
                recloss[m]=self.saes[m].ae[1].getErrorLoss(a[m][0],a[m][-1],recf)
            
#             ja=self.jsae.backward2Bottom(ja)
            jrecloss=self.jsae.ae[1].getErrorLoss(ja[0],ja[-1],recf)
        if diff_factor==0:
            diffgrad=None
        else:
#             print "diff is: ", diff_factor
            diffgrad=diff_factor*(ja[self.jdepth-1]-label)/self.batchsize
        g, jg =self.computeGrads2(a,ja,diffgrad, recf)
        return g, jg, recloss
      
    #Ehsan
    def trainOnePair(self, bat1, bat2, sim, epoch, recf, sim_diffcost, dis_diffcost):
        """
        trains one pair in which each element has two modalities
        im1: first element's image data
        tx1: first element's text data
        im2: second element's image data
        tx2: second element's text data
        sim: if the pair is in similar set
        recf: reconstruction factor
        """ 
        #consider diffcost?!
        a1 = []
        a2 = []
        for m in xrange(self.modalsCnt):
            a1.append(self.saes[m].forward2Top(bat1[m]))
            a2.append(self.saes[m].forward2Top(bat2[m])) 
        
        j1a = None
        j2a = None
        if self.has_joint:
            j1inp = gp.concatenate(tuple(e[self.depth-1] for e in a1),axis=1)
            j2inp = gp.concatenate(tuple(e[self.depth-1] for e in a2),axis=1)
            j1a = self.jsae.forward(j1inp)
            j2a = self.jsae.forward(j2inp)
        
        
        for m in xrange(self.modalsCnt):
            if self.has_joint:
                a1[m].append(j1a[-1][:,self.dims[m]:self.dims[m+1]])
            self.saes[m].backward2Bottom(a1[m])
            if self.has_joint:
                a2[m].append(j2a[-1][:,self.dims[m]:self.dims[m+1]])
            self.saes[m].backward2Bottom(a2[m])
        
#         j1a = j1a[1:-1]
#         j2a = j2a[1:-1]
        
        #get path grad for z
        #backpropagate x and y wrt z
        if self.has_joint:
            other1 = j2a[self.jdepth-1]
            other2 = j1a[self.jdepth-1]
        else:
            other1 = gp.concatenate(tuple(e[self.depth-1] for e in a2),axis=1)
            other2 = gp.concatenate(tuple(e[self.depth-1] for e in a1),axis=1)
        
        g1, jg1, rl1 = self.getSinglePathGrad2(a1, j1a, sim, other1, recf, sim_diffcost, dis_diffcost)
        g2, jg2, rl2 = self.getSinglePathGrad2(a2, j2a, sim, other2, recf, sim_diffcost, dis_diffcost)
        
        g = [[] for x in g1]
        for m in xrange(self.modalsCnt):
            g[m] = [[] for x in g1[m]]
            for i in xrange(len(g1[m])):
                g[m][i] = g1[m][i]+g2[m][i]
        
        jg = None
        if self.has_joint:        
            jg = [[] for x in jg1]
            for i in xrange(len(jg1)):
                jg[i] = jg1[i]+jg2[i]
        
        #this lines are just for debug:
        if self.has_joint:
            perf=[sim, self.getDiffLoss(j1a[self.jdepth-1],j2a[self.jdepth-1])]
        else:
            perf=[sim, self.getDiffLoss(gp.concatenate(tuple(e[self.depth-1] for e in a1),axis=1)
                                        ,gp.concatenate(tuple(e[self.depth-1] for e in a2),axis=1))]
#         for i in range(1,self.depth):
#             perf.append(self.getDiffLoss(ia[i],ta[i]))
#         a=ia[1:self.depth]+ta[1:self.depth]
#         ae=self.isae.ae[1:]+self.tsae.ae[1:]
#         for i in range(len(a)):
#             perf.append(ae[i].computeSparsity(a[i]))
        return np.array(perf), g, jg
    
    def trainClassifierOneBatch(self, trainbatch, labelbatch, epoch, diff_cost=1.0, recf=1.0):
        """
        trains one pair in which each element has two modalities
        im1: first element's image data
        tx1: first element's text data
        im2: second element's image data
        tx2: second element's text data
        sim: if the pair is in similar set
        recf: reconstruction factor
        """ 
        a = []
        
        for m in xrange(self.modalsCnt):
            a.append(self.saes[m].forward2Top(trainbatch[m])) 
        
        jinp = gp.concatenate(tuple(e[self.depth-1] for e in a),axis=1)
        ja = self.jsae.forward(jinp)
        
        for m in xrange(self.modalsCnt):
            a[m].append(ja[-1][:,self.dims[m]:self.dims[m+1]])
            self.saes[m].backward2Bottom(a[m])
        
        #get path grad for z
        #backpropagate x and y wrt z
        g, jg, rl = self.getClassificationGrad(a, ja, labelbatch, diff_factor=diff_cost, recf=recf)
        
        #this lines are just for debug:
        perfaf = gp.concatenate(tuple(e[0] for e in a),axis=1)
        perfal = gp.concatenate(tuple(e[-1] for e in a),axis=1)
        perf=self.getDiffLoss(perfaf, perfal)

        return perf, g, jg      
    
    def getMMReps(self, data):
        tops = []
        for i in range(self.modalsCnt):
            x = self.saes[i].forward2Top(data[i])
            tops.append(x[-1])
        if self.has_joint:
            jinp = gp.concatenate((tuple(tops)), axis=1)
            ja = self.jsae.forward2Top(jinp)
            return ja[-1].as_numpy_array()
        else:
            return gp.concatenate((tuple(tops)), axis=1).as_numpy_array()
            
        
        
    def getReps(self, imgData, txtData):
        """
        forward input data to top layer, then do sampling
        """
        ia=self.isae.forward2Top(imgData)
        ta=self.tsae.forward2Top(txtData)
        imgcode=ia[-1]
        txtcode=ta[-1]
        return imgcode.as_numpy_array(), txtcode.as_numpy_array()

    def getDiffLoss(self, x,y):
        loss=gp.sum((x-y)**2)*(0.5/x.shape[0])
        return loss

    #Ehsan
    def getNextParams(self):
        """Get next parameters"""
        rem = len(self.params) - self.paramInd
        if rem > 0:
            sdc = self.params[self.paramInd][0]
            ddc = self.params[self.paramInd][1]
            rc = self.params[self.paramInd][2]
        else :
            sdc = 0
            ddc = 0
            rc = 0
        self.paramInd += 1
        return rem, sdc, ddc, rc
        
    def checkPath(self, epoch):
        """get state info about fix which sae, adjust which sae"""
        epoch=epoch-self.epoch
        idx=self.statesIdx
        if epoch==self.states[idx][6]:
            self.epoch+=epoch
            idx=(self.statesIdx+1)%len(self.states)
            self.statesIdx=idx
            info=self.states[idx][0]
            print info
        k=1
        self.fix_img_path=self.states[idx][k]
        self.fix_txt_path=self.states[idx][k+1]
        imgcost=self.states[idx][k+2]
        txtcost=self.states[idx][k+3]
        diffcost=self.states[idx][k+4]
        return epoch,imgcost,txtcost,diffcost

    def doCheckpoint(self, outdir):
        """
        checkpoint for autoencoders along both two paths
        save them as 'modelcd' file under the same directory where the original model file locates
        """
        aes = []
        for m in xrange(self.modalsCnt):
            aes += self.saes[m].ae[1:]
#         aes=self.isae.ae[1:]+self.tsae.ae[1:]
        for ae in aes:
            path=os.path.join(outdir,ae.name)
            ae.save(path)
        super(MSAE,self).doCheckpoint(outdir)

    def extractValidationReps(self,imgData, txtData, reps_input_field,reps_output_field,outputPrefix=None):
        """evaluation data are small, thus stored in single file"""
        imgoutpath=self.readField(self.isae.ae[-1].config, self.isae.ae[-1].name, reps_output_field)
        txtoutpath=self.readField(self.tsae.ae[-1].config, self.tsae.ae[-1].name, reps_output_field)
        imgcode,txtcode=self.getReps(imgData, txtData)
        if not outputPrefix:
            np.save(imgoutpath,imgcode)
            np.save(txtoutpath,txtcode)
        else:
            np.save(outputPrefix+"img",imgcode)
            np.save(outputPrefix+"txt",txtcode)

    def extractTrainReps(self,imgDH, txtDH, numBatch):
        """training data may be large, thus use DataHandler to load them"""
        imgDH.reset()
        txtDH.reset()
        for i in range(numBatch):
            imgBatch=imgDH.getOneBatch()
            txtBatch=txtDH.getOneBatch()
            if imgBatch is None:
                break
            imgcode,txtcode=self.getReps(imgBatch, txtBatch)
            imgDH.write(imgcode)
            txtDH.write(txtcode)
        imgDH.flush()
        txtDH.flush()


    def getDisplayFields(self):
        s="neigbor dist(I->I,T->T,I->T,T->I),epoch , Img/Txt rec err,"
        formatt="%%%-ds, %%%-ds" %(7*(self.depth-1), 7*(self.depth-1))
        s+=formatt % ('layer-wise diff', '--img/txt layer-wise sparsity')
        return self.depth+1,self.depth*3-1,s

    #Ehsan
    def train(self):
        outputPrefix=self.readField(self.config,self.name,"output_directory")
        outputDir=os.path.join(outputPrefix,self.name)
        if not os.path.exists(outputDir):
            os.mkdir(outputDir)
        maxEpoch = int(self.readField(self.config, self.name, "max_epoch"))
        trainSize=int(self.readField(self.config, self.name, "train_size"))
        print "train size is: ", trainSize
        numBatch = int(trainSize / self.batchsize)
 
        normalizeImg=self.str2bool(self.readField(self.config, self.name, "normalize"))

        sim1 = []
        sim2 = []
        dis1 = []
        dis2 = []
        trainData = []
        valData = []
        queryData = []
        testData = []
        
        for i in xrange(self.modalsCnt):
            n = self.names[i]
            s = self.saes[i]
            
#             if self.readField(self.config, self.name,"extract_reps")=="True":
#                 output = self.readField(s.ae[-1].config, s.ae[-1].name, "train_reps")
#             else: 
            output = None
            
            t = self.readField(self.config, self.name, "sim"+n+"1")
            sim1.append(DataHandler(t, output, s.ae[1].vDim,
                                s.ae[-1].hDim, self.batchsize, numBatch))
            
            t = self.readField(self.config, self.name, "sim"+n+"2")
            sim2.append(DataHandler(t, output, s.ae[1].vDim,
                                s.ae[-1].hDim, self.batchsize, numBatch))
            
            t = self.readField(self.config, self.name, "dis"+n+"1")
            dis1.append(DataHandler(t, output, s.ae[1].vDim,
                                s.ae[-1].hDim, self.batchsize, numBatch))
            
            t = self.readField(self.config, self.name, "dis"+n+"2")
            dis2.append(DataHandler(t, output, s.ae[1].vDim,
                                s.ae[-1].hDim, self.batchsize, numBatch))

        showFreq = int(self.readField(self.config, self.name, "show_freq"))
        if showFreq > 0:
            visDir = os.path.join(outputDir, "vis")
            if not os.path.exists(visDir):
                os.makedirs(visDir)
            simpath = os.path.join(outputDir, "tmpVis", "sim");
            if not os.path.exists(simpath):
                os.makedirs(simpath)
            dispath = os.path.join(outputDir, "tmpVis", "dis");
            if not os.path.exists(dispath):
                os.makedirs(dispath)

        evalFreq = int(self.readField(self.config, self.name, "eval_freq"))
        if evalFreq!=0:
            qsize=int(self.readField(self.config, self.name, "query_size"))
            labelPath=self.readField(self.config,self.name,"val_label")
            label=np.load(labelPath)
            print "path: ", labelPath
            trainLabelPath=self.readField(self.config,self.name,"train_label")
            trainLabel=np.load(trainLabelPath)
            queryPath=self.readField(self.config, self.name, "query")
            
            for i in xrange(self.modalsCnt):
                n = self.names[i]
                s = self.saes[i]
                
                t = self.readField(s.ae[1].config, s.ae[1].name, "train_data")
                trainData.append(gp.garray(np.load(t)))
                
                t = self.readField(s.ae[1].config, s.ae[1].name, "validation_data")
                valData.append(gp.garray(np.load(t)))

        vallabelPath=self.readField(self.config,self.name,"val_label")
        vallabel=np.load(vallabelPath)
        testlabelPath=self.readField(self.config,self.name,"test_label")
        testlabel=np.load(testlabelPath)
        querylabelPath=self.readField(self.config,self.name,"query_label")
        querylabel=np.load(querylabelPath)
        
        for i in xrange(self.modalsCnt):
            n = self.names[i]
            s = self.saes[i]
            
            t = self.readField(s.ae[1].config, s.ae[1].name, "query_data")
            queryData.append(gp.garray(np.load(t)))
            t = self.readField(s.ae[1].config, s.ae[1].name, "test_data")
            testData.append(gp.garray(np.load(t)))
            
#         else:
#             print "Warning: no evaluation setting!"

        nCommon, nMetric, title=self.getDisplayFields()
        if self.verbose:
            print title
            
        print "params: ", len(self.params)
        rem, sdc, ddc, rc = self.getNextParams()
        while rem > 0 :
            print rem, sdc, ddc, rc
            rem, sdc, ddc, rc = self.getNextParams()
        self.paramInd = 0
        rem, sim_diffcost, dis_diffcost, reccost = self.getNextParams()
            
        while rem > 0 :                
            if evalFreq!=0:
                validation=evaluate.Evaluator(queryPath,vallabel,os.path.join(outputDir,'perf','val'), self.name, query_size=qsize,verbose=self.verbose)
                validation.setTrainLabel(vallabel)

            test=evaluate.Evaluator(queryPath,querylabel,os.path.join(outputDir,'perf','test'), self.name, query_size=qsize,verbose=self.verbose)
            test.setTrainLabel(testlabel)
            
            self.jdepth = self.ijdepth
            self.sparsityFactor = 0
            #pretrain
            self.trainClassifier()
            
            print 'testing pretrained model with parameters:', sim_diffcost, dis_diffcost, reccost
            ele=self.getMMReps(queryData)
            ele2=self.getMMReps(testData)
            test.evalSingleModal2(ele, ele2,maxEpoch, self.name, metric='euclidean')
            test.saveTarget(ele, ele2, metric='euclidean')
            
            for self.jdepth in xrange(self.ijdepth,self.max_jdepth+1):
                                
                self.sparsityFactor = 0   
                for epoch in range(maxEpoch):
                    print 'depth is: ', self.jdepth-1
                    for i in xrange(self.modalsCnt):
                        sim1[i].reset()
                        sim2[i].reset()
                        dis1[i].reset()
                        dis2[i].reset()
                    print "epoch: ", epoch
                    for i in range(numBatch):
                        
                        sim1batch = []
                        sim2batch = []
                        dis1batch = []
                        dis2batch = []
                        
                        for m in xrange(self.modalsCnt):
                            sim1batch.append(sim1[m].getOneBatch())
                            sim2batch.append(sim2[m].getOneBatch())
                            dis1batch.append(dis1[m].getOneBatch())
                            dis2batch.append(dis2[m].getOneBatch())
                        
                        #use imgcost and txt cost
                        curr, gs, jgs = self.trainOnePair(sim1batch, sim2batch, True, epoch, reccost, sim_diffcost, dis_diffcost)
                        curr2, gd, jgd = self.trainOnePair(dis1batch, dis2batch, False, epoch, reccost, sim_diffcost, dis_diffcost)
                        
                        g = [[] for x in gs]
                        for m in xrange(self.modalsCnt):
                            g[m] = [[] for x in gs[m]]
                            for i in xrange(len(gs[m])):
                                g[m][i] = gs[m][i]+gd[m][i]
                                
                        if self.has_joint:        
                            jg = [[] for x in jgs]
                            for i in xrange(len(jgs)):
                                jg[i] = jgs[i]+jgd[i]
                        
                        for m in xrange(self.modalsCnt):
                            self.saes[m].updateParams(epoch,g[m],self.saes[m].ae)
                        if self.has_joint:
                            self.jsae.updateParams(epoch,jg,self.jsae.ae)
    #                     perf=self.aggregatePerf(perf, curr)
    
    
                    if evalFreq!=0 and (1+epoch) % evalFreq == 0:
                        ele = self.getMMReps(valData)
                        ele2 = self.getMMReps(valData)
                        validation.evalSingleModal2(ele, ele2,epoch, self.name, metric='euclidean')
                
                if self.has_joint and self.jdepth < self.max_jdepth:
                    self.jsae.addAE()
                
#             if evalFreq != 0:
#                 test.saveMaps("maps-%d-%.3f-%.3f.npy" % (self.paramInd, diffcost, reccost))
                
            print 'testing model with parameters:', sim_diffcost, dis_diffcost, reccost
            ele=self.getMMReps(queryData)
            ele2=self.getMMReps(testData)
            test.evalSingleModal2(ele, ele2,maxEpoch, self.name, metric='euclidean')
            test.saveTarget(ele, ele2, metric='euclidean')
            
            ele = self.getMMReps(valData)
            ele2 = self.getMMReps(valData)
            validation.evalSingleModal2(ele, ele2,epoch, self.name, metric='euclidean')
            validation.saveTarget(ele, ele2, metric='euclidean')
#             
#             self.initsaes() 
        
            rem, sim_diffcost, dis_diffcost, reccost = self.getNextParams()
            print "ind is: ", self.paramInd


#         if self.readField(self.config, self.name, "checkpoint")=="True":
#             self.doCheckpoint(outputDir)

#         if self.readField(self.config, self.name,"extract_reps")=="True":
#             if evalFreq!=0:
#                 self.extractValidationReps(validateImgData, validateTxtData, "validation_data","validation_reps")
            #Uncomment this with new datahandlers
#             self.extractTrainReps(imgTrainDH, txtTrainDH, numBatch)

        self.saveConfig(outputDir)
        
    # This method can be used for training a classifier but here we use it for pre-training of the unfolded network
    def trainClassifier(self):
        print '-------------------------------'
        outputPrefix=self.readField(self.config,self.name,"output_directory")
        outputDir=os.path.join(outputPrefix,self.name)
        if not os.path.exists(outputDir):
            os.mkdir(outputDir)
        maxEpoch = int(self.readField(self.config, self.name, "max_pt_epoch"))
        trainSize=int(self.readField(self.config, self.name, "classifier_train_size"))
        numBatch = int(trainSize / (self.batchsize))
        
#         self.jsae.addAE(pretrain='mse')
        trainData=[]
        valData=[]
        testData=[]
        trainDH = []
        output = None
        
        for i in xrange(self.modalsCnt):
            n = self.names[i]
            s = self.saes[i]
            
            t = self.readField(s.ae[1].config, s.ae[1].name, "train_data")
            trainData.append(gp.garray(np.load(t)))
            
            t = self.readField(s.ae[1].config, s.ae[1].name, "validation_data")
            valData.append(gp.garray(np.load(t)))
            
            t = self.readField(s.ae[1].config, s.ae[1].name, "train_data")
            trainDH.append(DataHandler(t, output, s.ae[1].vDim,
                                s.ae[-1].hDim, self.batchsize, numBatch))
            
        t = self.readField(self.config, self.name, "train_label")
        cat_cnt = int(self.readField(self.config, self.name, "cat_cnt"))
        labelDH = DataHandler(t, output, cat_cnt,
                                   cat_cnt, self.batchsize, numBatch)
        
        evalFreq = int(self.readField(self.config, self.name, "eval_freq"))
        
        if evalFreq!=0:
            qsize=int(self.readField(self.config, self.name, "query_size"))
            labelPath=self.readField(self.config,self.name,"val_label")
            label=np.load(labelPath)
            print "path: ", labelPath
            trainLabelPath=self.readField(self.config,self.name,"train_label")
            trainLabel=np.load(trainLabelPath)
            queryPath=self.readField(self.config, self.name, "query")
            validation=evaluate.Evaluator(queryPath,label,os.path.join(outputDir,'perf'), self.name, query_size=qsize,verbose=self.verbose)
            validation.setTrainLabel(trainLabel)
            
        testlabelPath=self.readField(self.config,self.name,"test_label")
        testlabel=np.load(testlabelPath)
        print "path: ", testlabelPath
        for i in xrange(self.modalsCnt):
            n = self.names[i]
            s = self.saes[i]
            
            t = self.readField(s.ae[1].config, s.ae[1].name, "test_data")
            testData.append(gp.garray(np.load(t)))
        test=evaluate.Evaluator(queryPath,testlabel,os.path.join(outputDir,'perf'), self.name, query_size=qsize,verbose=self.verbose)
        test.setTrainLabel(trainLabel)

        print '>>>>>>>>>>>>>>>>>>>>>>pre-training the unfolded network<<<<<<<<<<<<<<<<<<<<'
        diff_cost = 0
        rec_cost = 0.1
        for epoch in range(maxEpoch):
            print 'depth is: ', self.jdepth-1
#             perf=np.zeros( nMetric)
            perf = 0
            for i in xrange(self.modalsCnt):
                trainDH[i].reset()
            labelDH.reset()
                
            print "epoch: ", epoch
            for i in range(numBatch):
                
                trainbatch = []
                
                for m in xrange(self.modalsCnt):
                    trainbatch.append(trainDH[m].getOneBatch())
                labelbatch = labelDH.getOneBatch()
                
#                 for m in xrange(self.modalsCnt):
#                     print trainbatch[m].shape
#                 print labelbatch
                
                #use imgcost and txt cost
                curr, g, jg = self.trainClassifierOneBatch(trainbatch, labelbatch, epoch, diff_cost=diff_cost, recf=rec_cost)
                perf += curr
                 
                for m in xrange(self.modalsCnt):
                    self.saes[m].updateParams(epoch,g[m],self.saes[m].ae, backprop=True)
                if self.has_joint:
                    self.jsae.updateParams(epoch,jg,self.jsae.ae, backprop=True)
  
#                 perf=self.aggregatePerf(perf, curr)

#             print 'perf is: ', perf
#             if evalFreq!=0 and (1+epoch) % evalFreq == 0:       
#                 ele=self.getMMReps(valData)
#                 validation.evalClassification(ele, label, epoch, self.name, metric='euclidean')
#         print 'test:'
#         ele=self.getMMReps(testData)
#         test.evalClassification(ele, testlabel, epoch, self.name, metric='euclidean')