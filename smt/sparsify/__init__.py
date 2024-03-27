import torch
import numpy as np

from ..sparsify.solvers import *
from ..sparsify.utils import *
from ..sparsify.dictionary_update import *

def learn_dictionary(data, n_dims=16, dictionary_size=2048, batch_size=128, steps=3_000_000, lambd=1.0):
    """
    Learn a dictionary of features from the data
    """
    xdim = ydim = n_dims #Patch size

    basis1 = torch.randn([xdim*ydim, dictionary_size]).cuda()
    basis1 = basis1.div_(basis1.norm(2,0))

    HessianDiag = torch.zeros(dictionary_size).cuda()
    ActL1 = torch.zeros(dictionary_size).cuda()
    signalEnergy = 0.
    noiseEnergy = 0.

    edgeBuff = 5
    spRange_t = data.shape[0]
    spRange_x = data.shape[1] - xdim - edgeBuff * 2
    spRange_y = data.shape[2] - ydim - edgeBuff * 2

    I = np.zeros([xdim*ydim, batch_size]).astype('float32')
    totalSteps1 = 0
    for i in range(totalSteps1, steps):
        for j in range(batch_size):
            xIdx = np.floor(np.random.rand()*spRange_x + edgeBuff).astype(int)
            yIdx = np.floor(np.random.rand()*spRange_y + edgeBuff).astype(int)
            sIdx = np.floor(np.random.rand()*spRange_t).astype(int)
            I[:,j] = data[sIdx,xIdx:xIdx+xdim,yIdx:yIdx+ydim].reshape([xdim*ydim])
        I_cuda = torch.from_numpy(I).cuda()

        ahat, Res = sparsify(I_cuda, basis1, method='positive-negative', return_error=True)

        #Statistics Collection (moving avarage updates)
        ActL1 = update_ActL1(ActL1, ahat)
        HessianDiag = update_HessianDiag(HessianDiag, ahat)
        signalEnergy, noiseEnergy = update_SNR(signalEnergy, noiseEnergy, I_cuda, Res)
        SNR = signalEnergy / noiseEnergy

        #Dictionary Update
        totalSteps1 = totalSteps1 + 1
        basis1 = quadraticBasisUpdate(basis1, Res, ahat, 0.001, HessianDiag, 0.1)

        #Print Information
        if i % 100 == 0:
            print(f"step: {int(totalSteps1)} | S-N-R: {SNR.item()} | "
                  f"Hessian min: {HessianDiag.min()}, max: {HessianDiag.max()} | "
                  f"Act min: {ActL1.min()}, max: {ActL1.max()}, sum: {ActL1.sum()}")

def plot_dictionary():
    """
    Plot the dictionary of features
    """
    pass

def sparsify(I, basis, method='positive-negative', return_error=False):
    """
    Sparsify the data
    """
    #Sparse Coefficients Inference by ISTA
    #For positive-only codes, use ISTA
    #For positive-negative codes, use ISTA_PN
    
    if method == 'positive-negative':
        ahat, Res = ISTA_PN(I, basis, 0.08, 250)
    elif method == 'positive-only':
        ahat, Res = ISTA(I, basis, 0.03, 1000)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    if return_error:
        return ahat, Res
    else:
        return ahat

