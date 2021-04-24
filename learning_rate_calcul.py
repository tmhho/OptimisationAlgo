def Wolfe_learning_rate(f, gradf, x, d, s0=5, eps1 = 1e-4, eps2 = 0.99, itermax = 20):
#     d : search direction
#     s0 : first approximation of learning rate
#     0 < esp1 < eps2 < 1
#     k := 0 ; s− = 0 ; s+ = +∞;

    iter, sn , sp = 0,0, 1e8
#     sn, sp are the minorant and majorant of learning rate s
    grad_x = gradf(x)
    f_x = f(x)
    s = s0 
    cond1 = f(x+s*d) <= f_x + eps1*s*grad_x.T.dot(d)
    cond2 = gradf(x+s*d).T.dot(d) >= eps2*grad_x.T.dot(d)
    while not cond1 or not cond2 :
        if not cond1: 
            sp = s
            s = (sn + sp)/2.
        elif not cond2:
            sn = s
            if sp < 1e8:
                s = (sn + sp)/2.
            else:
                s = 2*s 
        cond1 = f(x+s*d) <= f_x + eps1*s*grad_x.T.dot(d)
        cond2 = gradf(x+s*d).T.dot(d) >= eps2*grad_x.T.dot(d)
        iter += 1 
        if iter > itermax:
            break
    return s

from scipy.optimize import line_search
import numpy as np
def obj_func(x):
    return (x[0])**2+(x[1])**2
def obj_grad(x):
    return np.array([2*x[0], 2*x[1]])
start_point = np.array([1.8, 1.7])
search_gradient = np.array([-1.0, -1.0])
print("Scipy line_search : ", line_search(obj_func, obj_grad, start_point,search_gradient )[0])
print("Homemade line_search: ",Wolfe_learning_rate(obj_func,obj_grad,start_point, -obj_grad(start_point)))

def objective(x):
    return x[0]**2 + 4*x[1]**2
def grad(x):
    return 2*x*np.array([1, 4])
x = np.array([1., 3.]) #current point
p = -grad(x) #current search direction
a = line_search(objective, grad, x, p)[0]
print("Scipy line_search : ",a)
print("Homemade line_search: ",Wolfe_learning_rate(objective,grad,x, p))