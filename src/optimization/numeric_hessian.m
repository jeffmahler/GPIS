function hess_f = numeric_hessian(f, x, eps, n)
%HESSIAN Computes the hessian of a function f with n variables at a point x
    m = size(x,1);
    if nargin < 3
       eps = 1e-4;
    end
    if nargin < 4
       n = 1;
    end
    
    hess_f = zeros(m,m,n);
    for i = 1:m
       for j = 1:m
           if i == j
               x_plus = x;
               x_minus = x;
               x_plus(i,1) = x_plus(i,1) + eps;
               x_minus(i,1) = x_minus(i,1) - eps;
               hess_f(i, j,:) = (f(x_plus) - 2*f(x) + f(x_minus)) / (eps*eps);
           else
               xy_plus = x;
               xy_minus = x;
               x_plus = x;
               y_plus = x;
               xy_plus(i,1) = xy_plus(i,1) + eps;
               xy_plus(j,1) = xy_plus(j,1) + eps;
               xy_minus(i,1) = xy_minus(i,1) - eps;
               xy_minus(j,1) = xy_minus(j,1) - eps;
               x_plus(i,1) = x_plus(i,1) + eps;
               x_plus(j,1) = x_plus(j,1) - eps;
               y_plus(i,1) = y_plus(i,1) - eps;
               y_plus(j,1) = y_plus(j,1) + eps;
               hess_f(i, j,:) = (f(xy_plus) - f(x_plus) - f(y_plus) + f(xy_minus)) / (4*eps*eps);
           end
       end
    end
end


