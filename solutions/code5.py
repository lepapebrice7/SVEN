### Ecrivez votre code ici
# Si vous etes bloques, decommentez la premiere ligne et executez ce bloc.

def SVEN(X, y, t, l2):
    C = 0.5 / l2
    n, p = X.shape
    mat = (1/t) * np.dot(y, np.ones((1, p)))
    X_new = np.append(X - mat, X + mat, axis=1).T
    y_new = np.append(np.ones((p, 1)), -np.ones((p, 1)), axis=0)
    
    def max_vec(x1, x2):
        x_max = []
        for i in range(len(x1)):
            x_max += [max(x1[i, 0], x2[i, 0])]
        return np.array(x_max).reshape((-1, 1))
    
    if 2*p > n:
        w = SVMPrimal(X_new, y_new, C)
        zeta = np.ones((2*p, 1)) - y_new * np.dot(X_new, w)
        alpha = C * max_vec(np.zeros((2*p, 1)), zeta)
        
    else:
        alpha = SVMDual(X_new, y_new, C)
    
    beta = t * (alpha[:p]-alpha[p:])/np.sum(alpha)
    
    return beta