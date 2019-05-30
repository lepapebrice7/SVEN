### Ecrivez votre code ici
# Si vous etes bloques, decommentez la premiere ligne et executez ce bloc.

def SVMDual(X, y, C):
    m, q = X.shape
    Z = (X*y).T
    
    def func(alpha):
        Zalpha = np.sum(np.square(np.dot(Z, alpha)))
        return Zalpha + 1./(2*C) * np.sum(np.square(alpha)) - 2 * np.sum(alpha)
    
    def f_ieqcons(alpha):
        return alpha
    
    alpha0 = np.zeros(m)
    res = fmin_slsqp(func, alpha0, f_ieqcons=f_ieqcons)
    
    alpha = res.reshape((-1, 1))
    
    return alpha