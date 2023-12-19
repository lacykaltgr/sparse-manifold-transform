import torch

def get_trailing_eigenvectors(U, num_eigenvectors):
    """
    Get the trailing eigenvectors of a matrix U.

    Parameters:
    - U: The input matrix.
    - num_eigenvectors: Number of trailing eigenvectors to retrieve.

    Returns:
    - trailing_eigenvectors: Matrix containing the trailing eigenvectors.
    """
    # Perform eigenvalue decomposition
    eigenvalues, eigenvectors = torch.linalg.eig(U)

    # Sort eigenvalues and corresponding eigenvectors
    eigenvalues, indices = torch.sort(eigenvalues[:, 0])
    eigenvectors = eigenvectors[:, indices]

    # Select the trailing eigenvectors
    trailing_eigenvectors = eigenvectors[:, -num_eigenvectors:]

    return trailing_eigenvectors


def calculate_P(A, D, V):
    Q = torch.sqrt(V).inverse() @ A @ D @ D.T @ A.T @ torch.sqrt(V).inverse()
    U = get_trailing_eigenvectors(Q, 3)
    P =  U @ torch.sqrt(V).inverse()
    return P


# SGD solution
def learn_projection(A, D, V, n_epochs, lr, gamma, gamma_reg,):

    P = torch.rand(200, 2048).cuda() # initialize

    def shrinkage(A, P):
        # shrink each entry in the j-th column of P by gamma_reg * <a_j>
        # a_j is the j-th column of A
        # <.> denotes inner product
        pass

    for epoch in range(n_epochs):
        # whitened gradient computed on mini-batch
        P = -2 * gamma * lr * P @ A @ D @ D.T @ A.T @ V.sqrt().inverse()
        print("ok")
        #P = shrinkage(A, P) # regularization
        P = torch.sqrt(P @ V @ P.T).inverse() @ P # orthogonalization

    return P


