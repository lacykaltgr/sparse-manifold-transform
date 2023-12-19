import torch
from torch import nn

class SMT_layer(nn.Module):
    def __init__(self):
        super(SMT_layer, self).__init__()
        self.dictionary = None
        self.projection = None

    def forward(self, X):
        if self.dictionary is None:
            self.learn_dict(X)

        A = self.sparsify(X)
        if self.projection is None:
            self.learn_projection(A)
        B = self.project(A)
        return B

    # Sparsity functions
    def learn_dict(self, inputs):
        pass

    def plot_dict(self):
        pass

    def sparsify(self, X):
        pass

    # Manifold functions

    def learn_projection(self, A):
        pass

    def plot_projection(self):
        pass

    def project(self, A):
        pass


    