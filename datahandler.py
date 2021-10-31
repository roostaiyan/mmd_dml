import os
import sys
import numpy as np
import gnumpy as gp
import scipy.sparse as sp


def getBytes(mem_str):
    """Converts human-readable numbers to bytes.
       E.g., converts '2.1M' to 2.1 * 1024 * 1024 bytes.
    """
    unit = mem_str[-1]
    val = float(mem_str[:-1])
    if unit == 'G':
        val *= 1024*1024*1024
    elif unit == 'M':
        val *= 1024*1024
    elif unit == 'K':
        val *= 1024
    else:
        try:
            val = int(mem_str)
        except Exception:
                print '%s is not a valid way of writing memory size.' % mem_str
    return int(val)

def calcRowsForAlign(capacity, rows, ind):
    typesize=sys.getsizeof(float())
    ret=getBytes(capacity)/(rows*ind*typesize)
    return ret*rows

def LoadSparse(inputfile, verbose=False):
    """Loads a sparse matrix stored as npz file to its dense represent."""
    npzfile = np.load(inputfile)
    mat = sp.csr_matrix((npzfile['data'], npzfile['indices'],
                                  npzfile['indptr']),
                                  shape=tuple(list(npzfile['shape'])))
    if verbose:
      print 'Loaded sparse matrix from %s of shape %s' % (inputfile,
                                                          mat.shape.__str__())
    return mat.todense()


class DataHandler:
    """
    handler data operations: 
    read data from disk, cache it in mem and gpu mem; 
    write data to buffer and flush to disk
    """
    def __init__(self, inPath, outPath, indim, outdim, batchsize=100, numbatches=None, normalize=False,gpuMem='0.03G', mem='1G', writeBuffer='1G'):
        #GPUMem='3G', mem='15G', writeBuffer='1G'
        
        #init disk reader
        dr=DiskReader(inPath, indim)

        blocksize=calcRowsForAlign(gpuMem, batchsize,indim)
        #init memory cache
        memCache=MemCache(dr,blocksize,indim, mem)
        #init gpu cache
        gpuCache=GPUCache(memCache, batchsize, indim, gpuMem)

        self.cache=gpuCache
        
        #init disk writer
        if outPath is not None:
            self.dw=DiskWriter(outPath, outdim, batchsize, writeBuffer)

        self.numbatches=numbatches
        self.normalize=normalize
        self.cumbatches=0
        self.dim=indim
        
        if normalize:
            self.prepareStat(inPath)               

    def prepareStat(self, path):
        path=path.rstrip('/ ')
        stat_file=path+'_stat.npz'
        if os.path.exists(stat_file):
            stat=np.load(stat_file)
            self.mean=gp.as_garray(stat['mean'])
            self.std=gp.as_garray(stat['std'])
        else:
            self.mean, self.std=self.computeStat()
            np.savez(stat_file,mean=self.mean.as_numpy_array(), std=self.std.as_numpy_array())

    def computeStat(self):
        print 'Computing stats (mean and std)...'
        means=gp.zeros((self.numbatches, self.dim))
        variances=gp.zeros((self.numbatches, self.dim))
        i=0
        while True:
            batch=self.cache.getOneBatch()
            if batch==None:
                break
            print "fing i is :", i, " and batch mean is: ", batch.mean()
            means[i]=batch.mean(axis=0)
            variances[i]=gp.std(batch,axis=0)**2
            i+=1
        assert(i==self.numbatches)
        mean=means.mean(axis=0)
        std=gp.sqrt(variances.mean(axis=0)+gp.std(means,axis=0)**2)
        mean_std=std.mean()
        std+=(std==0.0)*mean_std
        self.reset()

        print 'Finish stats computing'
        return mean, std+1e-10

    def doNormalization(self, dat):
        dat-=self.mean
        dat/=self.std
        return dat
 
    def getOneBatch(self):
        batch=self.cache.getOneBatch()
        if batch==None: 
            assert(self.cumbatches==self.numbatches)
        else:
            self.cumbatches+=1
            if self.normalize:
                batch=self.doNormalization(batch)
        return batch

    def write(self, dat):
        self.dw.write(dat)

    def flush(self):
        self.dw.flush()

    def reset(self):
        self.cumbatches=0
        self.cache.reset()

class DiskReader:
    """
    read data from disk;
    path can be dir  or file path;
    read one file per call for func read
    """

    def __init__(self, path, ind):
        self.files=[]
        if os.path.isdir(path):
            files=os.listdir(path)
            files.sort()
            self.files=[os.path.join(path, f) for f in files]
        else:
            self.files=[path]
        #index for file to be read for next call of func read 
        self.fileIdx=0;
        self.dim=ind
        self.cumrows=0

    def read(self):
        """
        read one file from disk
        """
        if self.finished():
            return np.empty((0, self.dim))
        path=self.files[self.fileIdx]
        ext=os.path.splitext(path)[1]
        if  ext=='.npz':
            dat=LoadSparse(path)
        else:
            dat=np.load(path)
        #assert(dat.shape[1]==self.ind)
        self.fileIdx=self.fileIdx+1
        self.cumrows+=dat.shape[0]
        return dat

    def reset(self):
        self.fileIdx=0
        self.cumrows=0

    def finished(self):
        return self.fileIdx==len(self.files)


class MemCache:
    """
    memory cache
    cache data from DiskReader, will pass it to GPUCache
    """
    def __init__(self, diskReader, blocksize, ind, capacity):
        """
        blocksize: num of rows to pass to GPU Cache per read call
        capacity: mem cache size
        ind: feature dimension
        """
        self.typesize=sys.getsizeof(float())
        self.diskReader=diskReader
        self.blocksize=blocksize
        self.dim=ind
        self.capacity=getBytes(capacity)

        #store data read from disk reader
        self.data=np.empty((0,ind))
        #index, split pos for next call of func read
        self.index=0;
        #size in terms of rows
        self.size=0

        #show warning for only once
        self.showWarn=True
        
    def getOneBlock(self):
        end=self.index+self.blocksize
        if end<=self.size:
            ret=self.data[self.index:end]
            self.index=end
        else:
            ret=np.empty((self.blocksize,self.dim))
            retsize=self.size-self.index
            ret[0:retsize]=self.data[self.index:self.size]
            self.index=self.size
            while retsize<self.blocksize and not self.diskReader.finished():
                self.index=0
                self.data=self.diskReader.read()
                self.size=self.data.shape[0]
                if self.showWarn and self.size*self.typesize*self.dim>self.capacity:
                    print "Warning: file size is too large for mem cache!!"
                    self.showWarn=False
                min=self.blocksize-retsize
                if min>self.size:
                    min=self.size
                ret[retsize:retsize+min]=self.data[self.index:self.index+min]
                self.index+=min
                retsize+=min
            if retsize<self.blocksize:
                ret=ret[0:retsize]
                assert(self.diskReader.finished())
        
        return ret

    def reset(self):
        self.index=0
        self.size=0
        self.diskReader.reset()
        
class GPUCache:
    """
    load a block of data into gpu from mem
    gpu mem is algined with batchsize, i.e., maxrows % batchsize==0
    """
    def __init__(self, memCache, batchsize, ind, capacity):
        self.memCache=memCache
        self.batchsize=batchsize

        #read maxrows per call to MemCache, which is aligned to batchsize
        self.maxrows=calcRowsForAlign(capacity, batchsize, ind)

        self.data=gp.empty((self.maxrows, ind))
        self.index=0
        self.size=0

    def getOneBatch(self):
        """
        called to pass one batch data from gpu mem for processing
        """
        end=self.index+self.batchsize
        if end<=self.size:
            ret=self.data[self.index:end]
            self.index=end
        elif self.index==self.size:
            newdata=self.memCache.getOneBlock()
            self.size=int(newdata.shape[0]/self.batchsize)*self.batchsize
            if self.size<self.batchsize:
                return None
            self.index=0
            self.data[0:self.size]=gp.as_garray(newdata[0:self.size])
            ret=self.getOneBatch()
        else:
            raise Exception("GPU Mem is not algined with batchsize: %d, curr index %d, curr size %d" % (self.batchsize, self.index, self.data.shape[0]))
        return ret

    def reset(self):
        self.size=0
        self.index=0
        self.memCache.reset()

class DiskWriter:
    """
    write data to mem buffer, and flush to disk when full
    """
    def __init__(self, path, ind, batchsize=100, capacity='1G'):
        if os.path.exists(path):
            if os.path.isdir(path):
                files=os.listdir(path)
                if len(files)>0:
                    #print "the output directory is not empty"
                    for f in files:
                        os.remove(os.path.join(path,f))
            else:
                raise Exception("the output path is not directory")
        else:
            os.mkdir(path)
        self.dir=path
        #align buffer size with batchsize
        self.maxrows=calcRowsForAlign(capacity, batchsize, ind)

        self.data=None 
        self.index=0
        self.fileIdx=0

    def write(self, dat):
        """
        add dat to buffer
        """
        dat=gp.as_numpy_array(dat)
        end=self.index+dat.shape[0]
        if end<=self.maxrows:
            if self.data==None:
                self.data=np.empty((self.maxrows, dat.shape[1]))
            self.data[self.index:end]=dat
            self.index=end
        elif self.index==self.maxrows:
            self.flush()
            self.index=0
            self.write(dat)
        else:
            raise Exception("disk write buffer is not algined with batchsize")

    def flush(self):
        filePattern="data-%.5d" % self.fileIdx
        self.fileIdx+=1
        if self.index<self.maxrows:
            #debug
            data=self.data[0:self.index]
        else:
            data=self.data
        np.save(os.path.join(self.dir, filePattern), data)
