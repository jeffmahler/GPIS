function [x_grasp, x_all_iters] = find_antipodal_grasp_points(x_init, gpModel, ...
    cfg, gridDim, nu, lambda)
%FIND_ANTIPODAL_GRASP_POINTS Finds an antipodal set of grasp points

if nargin < 5
   nu = 1; 
end
if nargin < 6
   lambda = 0.05; 
end

% get dim
d = size(x_init,1) / 2;

% set up zeros functions (since we don't need them)
A_eq = 0;
b_eq = 0;
A_ineq = [-eye(2*d); eye(2*d)];
b_ineq = [zeros(2*d,1); gridDim*ones(2*d,1)];
Q = zeros(2*d, 2*d);
q = zeros(1, 2*d);

f = @(x) (100*det(gp_cov(gpModel, x(1:d,1)', [], true)) + ...
    100*nu*det(gp_cov(gpModel, x(d+1:2*d,1)', [], true)) - ...
    lambda*norm(x(1:d,1)' - x(d+1:2*d,1)'));
g = @(x) 0;
h = @(x) (surface_and_antipodality_functions(x, gpModel));

figure;
[x_grasp, x_all_iters] = penalty_sqp(x_init, Q, q, f, A_ineq, b_ineq, A_eq, b_eq, g, h, cfg);    

end

