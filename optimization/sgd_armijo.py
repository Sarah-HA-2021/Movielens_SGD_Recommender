import numpy as np 
def armijo_line_search(f, xk, pk, gfk, fxk, alpha0, shrink_factor=0.5,beta=1e-4, gamma=1.1):
  #Minimize over alpha, the function ``f(xₖ + αpₖ)``.
    # α > 0 is assumed to be a descent direction.
    #f : callable
        #Function to be minimized.
    #xk : array
        #Current point.
    #pk : array
        #Search direction.
    #gfk : array
        #Gradient of `f` at point `xk`.
    #fxk : float
        #Value of `f` at point `xk`.
    #alpha0 : scalar
        #Value of `alpha` at the start of the optimization.
    #shrink_factor : float, optional
        #Value of alpha shrinkage factor.
    #gamma : float, optional
        # value in the Armijo equation 
    #Returns
    #alpha : scalar
        #Value of `alpha` at the end of the optimization.

    dot_product = np.dot(gfk, pk)
    f_eval = f(xk + alpha0*pk)
    
    while not f_eval <= fxk+ beta*alpha0*dot_product:
        alpha0 = alpha0 * shrink_factor
        f_eval = f(xk + alpha0*pk)
    return alpha0


def gradient_descent_with_armijo(f, x0, gamma=1, n_iterations=100, eps=10-6,beta=1e-4,alpha=1.5,shrink_factor=0.3):
  x_cur= x0
  for i in range(n_iterations+1):
     value1= f(x_cur)
     #aproxomation 
     value2= (f(x_cur+ eps/2) - f(x_cur-eps/2)) / eps # 
     alpha=armijo_line_search(f, x_cur, -1*value2, value2, value1,alpha,shrink_factor,beta,gamma)
     new_x = x_cur - alpha * value2
     x_cur = new_x
  return f(x_cur) ,"at point", x_cur