function grad_f = numeric_gradient(f, x, eps, n)
%GRADIENT Computes the gradient of a function f with n variables at a point x
    m = size(x,1);
    if nargin < 3
       eps = 1e-4;
    end
    if nargin < 4
       n = 1;
    end
    
    grad_f = zeros(m,n);
    for i = 1:m
       x_plus = x;
       x_minus = x;
       x_plus(i,1) = x_plus(i,1) + eps;
       x_minus(i,1) = x_minus(i,1) - eps;
       grad_f(i,:) = (f(x_plus) - f(x_minus)) / (2*eps);
    end
end

