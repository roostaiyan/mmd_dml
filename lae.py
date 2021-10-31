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
from ae import AE
import gnumpy as gp


class LAE(AE):
    """
    autoencoder with liner decoder
    to model visible layer of Gaussian distribution, e.g., natural images
    a1=logistic(vW+b)
    a2=a1W2+b
    """

    def __init__(self, config, name):
        super(LAE, self).__init__(config, name)

    def backwardOneStep(self, a):
        z=gp.dot(a,self.W2)+self.b2
        return z

    def computeDlast(self, a0,a2,factor=1.0):
        d3=(a2-a0)*factor
        return d3
