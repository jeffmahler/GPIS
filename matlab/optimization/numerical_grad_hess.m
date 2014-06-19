function [grad,hess] = numerical_grad_hess(f,x, full_hessian)
% numerical gradient and diagonal hessian

y = f(x);
assert(length(y)==1);

grad = zeros(1, length(x));
hess = zeros(length(x));

eps = 1e-5;
xp = x;


if nargout > 1
    if ~full_hessian
        for i=1:length(x)
            xp(i) = x(i) + eps/2;
            yhi = f(xp);
            xp(i) = x(i) - eps/2;
            ylo = f(xp);
            xp(i) = x(i);
            hess(i,i) = (yhi + ylo - 2*y)/(eps.^2 / 4);
            grad(i) = (yhi - ylo) / eps;
        end
    else
        grad = numerical_jac(f,x);
        hess = numerical_jac(@(x) numerical_jac(f,x), x);
        hess = (hess + hess')/2;
    end
end
    

end