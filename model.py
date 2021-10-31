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
import os
import cPickle as pickle
from evaluate import Evaluator
import gnumpy as gp
import ConfigParser
from datahandler import DataHandler

class Model(object):
    """
    base class
    matrix are in row fashion, i.e., one row per case
    """

    def __init__(self, config, name):
        self.config = config

        #print intermediate info
        if config.has_option(name,'verbose'):
            self.verbose=config.getboolean(name,'verbose')
        else:
            self.verbose=config.getboolean('DEFAULT','verbose')

        #here debug means do numeric gradient check
        self.debug=self.str2bool(self.readField(config, name, "debug"))

        #model name, i.e., the section in config file
        self.name = name

        #sections in config file related to this autoencoder
        self.sections=[name]

        self.batchsize = int(self.readField(config, name, "batchsize"))
        self.trainDataLoader = None

    def aggregatePerf(self, prev, curr, r=0.9):
        """
        aggregate performance of batches with exponetial decay
        """
        return r*prev+curr*(1-r)

    def doCheckpoint(self, outdir):
        checkpointPath = os.path.join(outdir, 'model')
        self.save(checkpointPath)

    def saveConfig(self, outdir):
        """
        save configures of this autoencoder, all other sections/options are removed
        """
        sections=self.config.sections()
        for section in sections:
            if section not in self.sections:
                self.config.remove_section(section)
        configfile=os.path.join(outdir, 'config.ini')
        with open(configfile, 'wb') as fp:
            self.config.write(fp)

    @staticmethod
    def load(path):
        """
        load model from disk, hyper-params can be reload from config file
        or just use the previous settings.
        """
        file = open(path)
        rbm = pickle.load(file)
        return rbm

    def save(self, path):
        #save model to disk
        file = open(path, "wb")
        pickle.dump(self, file)

    def trainOneBatch(self, input, epoch, computeStat=True):
        pass

    def getReps(self, v):
        pass

    def extractValidationReps(self, dat, output_path): 
        """
        extract representations of input data, i.e., top layer vector
        dat may be the option name for the path of input data
        save it to disk, location is read from config
        """
        dat=gp.as_garray(dat)
        reps=self.getReps(dat)
        np.save(output_path, reps)

    def extractTrainReps(self, datahandler, numBatch):
        """
        extract representations for (big) training data through DataHandler
        """
        for tl in datahandler:
                tl.reset()
            
        for i in range(numBatch):
            batches = [None for x in datahandler] 
            for i in range(len(batches)):
                batches[i] = datahandler[i].getOneBatch()
            
            batch = gp.concatenate(tuple(batches), axis=1)
            if batch is None:
                break
            reps=self.getReps(batch)
            datahandler[0].write(reps)
        datahandler[0].flush()

    def computeIncRatio(self,w,inc=None):
        """
        compute weight increment ratio
        """
        if inc==None:
            return self.incW1.euclid_norm()/w
        else:
            return inc.euclid_norm()/w

    def readField(self, config, section, option):
        try:
            ret = config.get(section, option)
        except ConfigParser.NoOptionError:
            ret = 0
        except ConfigParser.NoSectionError:
            ret = 0
        if ret == 0:
            ret = config.get("DEFAULT", option)
            config.set(section, option, ret)
        if self.verbose:
            print "%s=%s" % (option,ret)
        return ret

    def readLearningRate(self,config, sectionName):
        epsilon = self.readField(config,sectionName, "base_learning_rate")
        epsDecayHalfEpochs = self.readField(config,sectionName, "learning_rate_decay_half_epochs")
        return (float(epsilon), int(epsDecayHalfEpochs))

    def readMomentum(self, config, sectionName):
        momentumStart = self.readField(config,sectionName, "start_momentum")
        momentumEnd = self.readField(config,sectionName, "end_momentum")
        momentumDecayEpochs = self.readField(config,sectionName, "momentum_decay_epochs")
        return (float(momentumStart), float(momentumEnd), int(momentumDecayEpochs))

    def softmax(self, x):
        max=gp.max(x,axis=1)
        x=x-max[:,gp.newaxis]
        y=gp.exp(x)
        s=gp.sum(y,1)
        z=y/s[:,gp.newaxis]
        return z


    def loadModel(self, config, name):
        """
        name: path to model file or section name for the model
        """
        if os.path.exists(name):
            from ae import AE
            model=AE.load(name)
        else:
            modelname=self.readField(config, name, "model")
            if modelname=="lae":
                from lae import LAE
                model=LAE(config, name)
            elif modelname=="pae":
                from pae import PAE
                model=PAE(config, name)
            elif modelname=='ae':
                from ae import AE
                model=AE(config, name)
        return model

    def vectorParam(self,params):
        """:
        params: an array of parammeter (W1,b1,[W2,b2]) of autoencoders from bottom to top of a single path
        """
        n=0
        for param in params:
            n+=param.size
        p=gp.zeros(n)
        k=0
        for param in params:
            s=param.size
            p[k:k+s]=param.reshape(-1)
            k+=s
        return p

    def getDisplayFields(self):
        pass

    def printEpochInfo(self,epoch, perf, m):
        """
        m: # of common metric for multiple paths
        """
        #n is # of model specific metrics
        n=(len(perf)-m)/2
        info="epoch %2d: " % epoch
        for i in range(m):
            info+="%.4f, " % perf[i]
        for i in range(2):
            info+="--("
            offset=m+i*n
            for i in range(n):
                info+="%.4f, " % perf[i+offset]
            info+=")"
        print info

    def str2bool(self, s):
        if s.strip() in ['True','1','Yes','true', 'yes']:
            return True
        else:
            return False

    def train(self):
        outputPrefix=self.readField(self.config,self.name,"output_directory")
        outputDir=os.path.join(outputPrefix,self.name)
        
        if not os.path.exists(outputDir):
            os.makedirs(outputDir)

        showFreq = int(self.readField(self.config, self.name, "show_freq"))
        if showFreq > 0:
            visDir = os.path.join(outputDir,'vis')
            if not os.path.exists(visDir):
                os.mkdir(visDir)
        #do normalization for images if they are not normalized before
        normalize=self.str2bool(self.readField(self.config, self.name, "normalize"))
        trainDataSize=int(self.readField(self.config, self.name, "train_size"))
        numBatch = trainDataSize / self.batchsize
        
        
        if self.readField(self.config,self.name,"extract_reps")=="True":
            trainRepsPath=self.readField(self.config, self.name, "train_reps")
        else:
            trainRepsPath=None
        print trainDataSize
        
        #Ehsan
        trainDataPath = (self.readField(self.config,self.name,'train_data')).split(',')
        print trainDataPath
        trainDataLoader =  [None for x in trainDataPath]
        trainDataFiles=len(trainDataPath)
        dims = [self.vDim]
        if self.config.has_option(self.name, 'train_dims'):
            dimsstr = (self.readField(self.config,self.name,'train_dims')).split(',')
            dims = [int(i) for i in dimsstr]
        for i in range(trainDataFiles):
            trainDataLoader[i]=DataHandler(trainDataPath[i], trainRepsPath, dims[i], self.hDim, self.batchsize,numBatch, normalize)
        
        evalFreq=int(self.readField(self.config,self.name,'eval_freq'))
        if evalFreq!=0:
            qsize=int(self.readField(self.config, self.name, "query_size"))
            evalPath=self.readField(self.config,self.name,"validation_data")
            labelPath=self.readField(self.config,self.name,"label")
            queryPath=self.readField(self.config, self.name, "query")
            label=np.load(labelPath)
            eval=Evaluator(queryPath,label ,os.path.join(outputDir,'perf'), self.name, query_size=qsize,verbose=self.verbose)
            validation_data=gp.garray(np.load(evalPath))
            if normalize:
                validation_data=trainDataLoader.doNormalization(validation_data)

        maxEpoch = int(self.readField(self.config, self.name, "max_epoch"))

        nCommon, nMetric, title=self.getDisplayFields()
        if self.verbose:
            print title
        for epoch in range(maxEpoch):
            perf=np.zeros( nMetric)
            for tl in trainDataLoader:
                tl.reset()
            
            for i in range(numBatch):
                batches = [None for x in trainDataLoader] 
                for i in range(len(batches)):
                    batches[i] = trainDataLoader[i].getOneBatch()
                
                batch = gp.concatenate(tuple(batches), axis=1)
                curr = self.trainOneBatch(batch, epoch, computeStat=True)
                perf=self.aggregatePerf(perf, curr)

            if showFreq != 0 and (1+epoch) % showFreq == 0:
                validation_code=self.getReps(validation_data)
                np.save(os.path.join(visDir, '%dvis' % (1+epoch)), validation_code)
            if evalFreq !=0 and (1+epoch) % evalFreq ==0:
                validation_code=self.getReps(validation_data)
                eval.evalSingleModal(validation_code,epoch,self.name+'V')
                validation_code=None
            if self.verbose:
                self.printEpochInfo(epoch,perf,nCommon)

        #Ehsan
        try:
            keepDL=self.str2bool(self.readField(self.config, self.name, "keep_dataloader"))    
            if keepDL:
                print 'saving tdl for ', self.name
                self.trainDataLoader=trainDataLoader
        except:
            print 'exception occured'
            
        if self.readField(self.config,self.name,"checkpoint")=="True":
            self.doCheckpoint(outputDir)

        if self.readField(self.config,self.name,"extract_reps")=="True":
            if evalFreq!=0:
                validation_reps_path=self.readField(self.config, self.name, "validation_reps")
                self.extractValidationReps(validation_data, validation_reps_path)
            self.extractTrainReps(trainDataLoader, numBatch)
        
        self.saveConfig(outputDir)
