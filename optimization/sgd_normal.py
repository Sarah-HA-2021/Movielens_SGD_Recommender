def normal_gradient_descent(f, x0, alpha ,n_iterations=100, eps=10-6):
  # implement n_iterations of the gradient descent
  # start at point x0
  # at every step of gradient descent take alpha that satisfies
  # the Armijo’s constraints for given beta and gamma
  # return the point of minimum
  cur_x = x0
  #Armijo condition 
  
  for i in range(n_iterations+1):
    # df_approx = (f(cur_x + eps) - f(cur_x)) / eps
    df_approx = (f(cur_x + eps/2) - f(cur_x-eps/2)) / eps
    new_x = cur_x - alpha* df_approx
    cur_x = new_x
  return f(cur_x)