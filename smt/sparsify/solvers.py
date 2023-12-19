# Yubei Chen, Sparse Manifold Transform Lib Ver 0.1
"""
This file contains multiple method to sparsify the coefficients
"""
import time
import numpy as np
import numpy.linalg as la
import torch
import torch.nn.functional as F



def ISTA_PN(I,basis,lambd,num_iter,eta=None):
    # This is a positive-negative PyTorch-Ver ISTA solver
    
    batch_size=I.size(1)
    M = basis.size(1)
    
    if eta is None:
        L = torch.max(torch.linalg.eigvalsh(torch.mm(basis, basis.t()), UPLO='U'))
        eta = 1./L

    Res = torch.zeros(I.size(), device='cuda')
    ahat = torch.zeros(M, batch_size, device='cuda')

    for t in range(num_iter):
        ahat = ahat.add(eta * basis.t().mm(Res))
        ahat_sign = torch.sign(ahat)
        ahat.abs_()
        ahat.sub_(eta * lambd).clamp_(min = 0.)
        ahat.mul_(ahat_sign)
        Res = I - torch.mm(basis,ahat)
    return ahat, Res


def FISTA(I, basis, lambd, num_iter, eta=None):
    # This is a positive-only PyTorch-Ver FISTA solver
    
    batch_size=I.size(1)
    M = basis.size(1)
    if eta is None:
        L = torch.max(torch.linalg.eigvalsh(torch.mm(basis, basis.t()), UPLA='U'))
        eta = 1./L

    tk_n = 1.
    tk = 1.
    Res = torch.zeros(I.size(), device='cuda')
    ahat = torch.zeros(M,batch_size, device='cuda')
    ahat_y = torch.zeros(M,batch_size, device='cuda')

    for t in range(num_iter):
        tk = tk_n
        tk_n = (1+np.sqrt(1+4*tk**2))/2
        ahat_pre = ahat
        Res = I - torch.mm(basis,ahat_y)
        ahat_y = ahat_y.add(eta * basis.t().mm(Res))
        ahat = ahat_y.sub(eta * lambd).clamp(min = 0.)
        ahat_y = ahat.add(ahat.sub(ahat_pre).mul((tk-1)/(tk_n)))
    Res = I - torch.mm(basis,ahat)
    return ahat, Res

def ISTA(I, basis, lambd, num_iter, eta=None):
    # This is a positive-only PyTorch-Ver ISTA solver

    batch_size=I.size(1)
    M = basis.size(1)
    if eta is None:
        L = torch.max(torch.linalg.eigvalsh(torch.mm(basis,basis.t()), UPLA='U')[0])
        eta = 1./L

    Res = torch.zeros(I.size(), device='cuda')
    ahat = torch.zeros(M,batch_size, device='cuda')

    for t in range(num_iter):
        ahat = ahat.add(eta * basis.t().mm(Res))
        ahat = ahat.sub(eta * lambd).clamp(min = 0.)
        Res = I - torch.mm(basis,ahat)
    return ahat, Res



