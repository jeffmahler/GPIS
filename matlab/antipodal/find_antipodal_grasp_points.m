function [x_grasp, x_all_iters] = find_antipodal_grasp_points(x_init, gpModel, ...
    cfg, gridDim, com)
%FIND_ANTIPODAL_GRASP_POINTS Finds an antipodal set of grasp points

use_com = true;
if nargin < 5
    use_com = false;
end

nu = cfg.nu;
lambda = cfg.lambda;

% get dim
d = size(x_init,1) / 2;

% set up zeros functions (since we don't need them)
A_eq = 0;
b_eq = 0;
A_ineq = [-eye(2*d); eye(2*d)];
b_ineq = [zeros(2*d,1); gridDim*ones(2*d,1)];
Q = zeros(2*d, 2*d);
q = zeros(1, 2*d);

f = @(x) (antipodality_penalty(x, gpModel, nu, lambda, com));
if use_com
    g = @(x) (friction_cone_constraint(x, gpModel, cfg.fric_coef, com, cfg.com_tol));
    h = @(x) (surface_and_antipodality_functions(x, gpModel));%, com));
else
    g = @(x) (friction_cone_constraint(x, gpModel, cfg.fric_coef));
    h = @(x) (surface_and_antipodality_functions(x, gpModel));
end

[x_grasp, x_all_iters] = penalty_sqp(x_init, Q, q, f, A_ineq, b_ineq, A_eq, b_eq, g, h, cfg);    

end

