### Ecrivez votre code ici
# Si vous etes bloques, decommentez la premiere ligne et executez ce bloc.

def SVMPrimal(X, y, C):
    m, q = X.shape
    
    def func(x):
        w = x[:q]
        zeta = x[q:]
        return C * np.sum(zeta) + 0.5 * np.sum(np.square(w))
    
    def f_ieqcons(x):
        w = x[:q]
        zeta = x[q:]
        cons = y.reshape((1, -1))[0] * np.dot(X, w) + zeta - 1
        return np.append(cons, zeta)
    
    x0 = np.zeros(q+m)
    res = fmin_slsqp(func, x0, f_ieqcons=f_ieqcons)
    
    return res[:q].reshape((q, 1))