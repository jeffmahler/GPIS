function t = btls(f, grad_f, x, v, alpha, beta)
% v - direction of descent
% x - point at which to evaluate
% alpha - between 0 and 1/2
% beta - between 0 and 1
    t = 1;
    while f(x + t*v) > f(x) + alpha*t*grad_f'*v
       t = beta*t; 
    end
end

