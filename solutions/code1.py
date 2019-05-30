### Ecrivez votre code ici
# Si vous etes bloques, decommentez la premiere ligne et executez ce bloc.

def LASSO(X, y, t):
    
    def func(x):
        beta = x.reshape((-1, 1))
        return np.sum(np.square(y - np.dot(X, beta)))
    
    def f_ieqcons(x):
        return t - np.sum(np.abs(x))
    
    p = X.shape[1]
    x0 = np.zeros(p)
    res = fmin_slsqp(func, x0, f_ieqcons=f_ieqcons)
    
    return res.reshape((p, 1))