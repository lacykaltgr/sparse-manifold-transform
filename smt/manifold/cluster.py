import torch


def natural_nghb_matrix(A, scheme='temporal', spatial_range=3, spatial_weight = 1,
                        temporal_range=1, temporal_causality=False, temporal_weight = 1):
    dims = A.shape[0]
    n_patches = A.shape[1]
    offset = 121
    NB = torch.zeros(n_patches, n_patches) # initialize
    NB.fill_diagonal_(1) # set diagonal (i, i) to 1

    if scheme == 'temporal' or scheme == 'spatiotemporal':
        # A: (sparse dimensions, n data points)
        # find the previous (and the next if temporal_causality=False) sequence elements for the patch
        # get them in ((x, y), weight->delta_t) format
        for i in range(n_patches):
            for t in range(temporal_range):
                backward = i - (t+1) * offset
                if backward > 0:
                    NB[i, backward] = -0.5

                if not temporal_causality:
                    forward = i + (t+1) * offset
                    if forward < n_patches:
                        NB[i, forward] = -0.5
        return NB.cuda()

        # if spatiotemporal multiply by temporal_weight
        # set (x, y) of NB

    if scheme == 'spatial' or scheme == 'spatiotemporal':
        raise NotImplementedError()
        # find the neighbouring pathces spatially
        # get them in ((x, y), distance) format
        # if spatiotemporal multiply by spatial_weight
        # set (x, y) of NB