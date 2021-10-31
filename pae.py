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
from ae import AE


class PAE(AE):
    """
    autoencoder with possion decoder
    p(vi=xi|zi)=ps(xi,ci)
    ci=N*zi/sum(zi)
    zi=a*W+b
    ps(n,c)=e^(-c)c^n/n!
    loss J=\prod p(vi=xi|zi)
    dJ/dzi=ci-xi
    """

    def __init__(self, config, name):
        super(PAE,self).__init__(config, name)
        self.factor=gp.ones(1000)
        for i in range (1,self.factor.size):
            self.factor[i]=self.factor[i-1]*i
        self.N=None

    def forwardOneStep(self,a):
        self.N=gp.as_garray(gp.sum(a,axis=1).as_numpy_array().astype(int))
        z=gp.dot(a,self.W1)+self.b1
        return gp.tanh(z)

    def backwardOneStep(self,a):
        z=gp.dot(a,self.W2)+self.b2
        c=self.softmax(z)
        a2=c*self.N[:,gp.newaxis]
        return a2

    def getErrorLoss(self, a0, a2,factor=1.0):
        """
        error is measured by neg log likelihood
        """
        poww=a2**a0
        p=gp.exp(-a2)*poww/self.factor[a0] 
        l=gp.log(p)
        ret = -l.sum(axis=1).mean()*factor
        return ret

    def computeDlast(self,a0,a2,factor=1.0):
        return (a2-a0)*factor
