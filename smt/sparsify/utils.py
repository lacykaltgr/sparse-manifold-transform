import torch

def update_ActL1(ActL1, ahat, ACT_HISTORY_LEN=300):
    ActL1 = ActL1.mul((ACT_HISTORY_LEN-1.0)/ACT_HISTORY_LEN) + ahat.abs().mean(1)/ACT_HISTORY_LEN
    return ActL1
    
def update_HessianDiag(HessianDiag, ahat, ACT_HISTORY_LEN=300):
    HessianDiag = HessianDiag.mul((ACT_HISTORY_LEN-1.0)/ACT_HISTORY_LEN) + torch.pow(ahat,2).mean(1)/ACT_HISTORY_LEN
    return HessianDiag
    
def update_SNR(S, N, I, Res, ACT_HISTORY_LEN=300):
    decay = (ACT_HISTORY_LEN-1.0)/ACT_HISTORY_LEN
    signalEnergy= S * decay + torch.pow(I,  2).sum()/ACT_HISTORY_LEN
    noiseEnergy = N * decay + torch.pow(Res,2).sum()/ACT_HISTORY_LEN
    return signalEnergy, noiseEnergy
    
    