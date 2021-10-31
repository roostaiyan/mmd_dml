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
import matplotlib.pyplot as plt
import sys
import matplotlib
import argparse
import os
from matplotlib.lines import lineStyles

matplotlib.rcParams.update({'font.size': 22})
matplotlib.rc('lines', linewidth=2)

def plotCrossPairs(simg, stxt, plt,n,m):
    for i in range(m):
        vertices = np.zeros((2,2))
        vertices[0]=simg[i]
        vertices[1]=stxt[i]         
        plt.plot(vertices[:,0], vertices[:,1],c='r')
    cImg=['b']*n
    cTxt=['w']*n
    plt.scatter(simg[0:n,0],simg[0:n,1],c=cImg[0:n],s=150)  
    plt.scatter(stxt[0:n,0],stxt[0:n,1],c=cTxt[0:n],s=150)          


def plot2D(imgpath, txtpath, outpath,idx, n,m):
    simg=np.load(imgpath)
    stxt=np.load(txtpath)
    simg=simg[idx]
    stxt=stxt[idx]

    plotCrossPairs(simg, stxt, plt, n,m)
    plt.ylim(0,1)
    plt.xlim(0,1)
    plt.savefig(outpath)
    #plt.show()


def drawMAP(dat, outputpath,legendsize=28):             
    plt.clf()
    tasks=[r'$\mathbb{Q}_{I\rightarrow I}$',r'$\mathbb{Q}_{T\rightarrow T}$', r'$\mathbb{Q}_{I\rightarrow T}$', r'$\mathbb{Q}_{T\rightarrow I}$']
    dat=np.asarray(dat,dtype=np.float32)
    print dat.shape, dat.dtype
    x=np.arange(0,dat.shape[1],3)   
    x=np.append(x,[dat.shape[1]-1])
    #print x
    lines=[]
    style=['o-','^-','s-','D-']
    for i in range(4):
        line,=plt.plot(x+1,dat[i][x],style[i],ms=10)
        lines.append(line)
    for i in range(dat.shape[0]):
        if dat[i][-1]<0.4:
            legendsize=20
    plt.legend(lines, tasks, loc='lower right', prop={'size':legendsize})
    plt.xlim(0,dat.shape[1]+2)
    plt.ylim(0.1,0.55)
    plt.xlabel(r'epoch (time)')
    plt.ylabel('MAP')
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig(outputpath,bbox_inches='tight')
    
#Ehsan
def myPlotMap(dat, outputpath):
    plt.clf()
    plt.plot(dat, '-^', ms=7)
    plt.xlim(0,dat.shape[0]+1)
    plt.ylim(0,1)
    plt.xlabel(r'epoch (time)')
    plt.ylabel('MAP')
    titl = os.path.basename(outputpath)
    titl, ext = os.path.splitext(titl) 
    plt.title(titl)
    plt.savefig(outputpath,bbox_inches='tight')
    
if __name__=="__main__":
    parser=argparse.ArgumentParser(description="convert between mat and npy")   
    parser.add_argument("input", help="input dir: img.npy,txt.npy;")        
    parser.add_argument("index", help="path of sample index")       
    parser.add_argument("output", help="outpu dir") 
    parser.add_argument("-f", default=".png", choices=['.png','.eps'], help="format:.png or .eps")
    parser.add_argument("-n", type=int, default=300, help="sampl size")
    parser.add_argument("-m", type=int, default=25, help="# of pairs to be collected")
    parser.add_argument("-s", type=int, help="dataset size, needed to generate random sample index")

    args=parser.parse_args()

    datadir=args.input
    outputdir=args.output


    if os.path.exists(args.index):
        idx=np.load(args.index)
    else:
        if not args.s:
            print "dataset size must be provided for sampling index"
            sys.exit(0)
        print "argss: ", args.s
        idx=np.random.randint(0,args.s,args.n)
        print "maxidx: ", np.max(idx)
        np.save(args.index,idx)

    files=os.listdir(datadir)
    imgfs=[f for f in files if "img" in f]
    txtfs=[f for f in files if "txt" in f]

    imgfs.sort()
    imgfs.sort(key=len)

    txtfs.sort()
    txtfs.sort(key=len)

    i=0
    for (img,txt) in zip(imgfs, txtfs):     
        if i<10:
            name='00'+str(i)
        elif i<100:
            name='0'+str(i)
        else:
            name=str(i)

        imgdatapath=os.path.join(datadir,img)
        txtdatapath=os.path.join(datadir,txt)
        vispath=os.path.join(outputdir,name+args.f)
        if i==0:
            plot2D(imgdatapath,txtdatapath,os.path.join(outputdir, "scatter.png"),idx,args.n,0) 
            plt.clf()                       

        plot2D(imgdatapath,txtdatapath,vispath,idx,args.m,args.m)                   
        #sys.exit(0)
        i+=1
        plt.clf()
