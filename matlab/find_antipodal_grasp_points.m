function x_grasp = find_antipodal_grasp_points(x_init, gpModel, ...
    surfaceImage, gridDim, scale, nu)
%FIND_ANTIPODAL_GRASP_POINTS Finds an antipodal set of grasp points

if nargin < 5
   scale = 1; 
end
if nargin < 6
   nu = 1; 
end

% Set parameters of the optimizer
cfg = struct();
cfg.max_merit_coeff_increases = 5;
cfg.merit_coeff_increase_ratio = 20;
cfg.initial_penalty_coeff = 1.;
cfg.initial_trust_box_size = 50;
cfg.trust_shrink_ratio = .9;
cfg.trust_expand_ratio = 2.0;
cfg.min_approx_improve = 1e-8;
cfg.min_trust_box_size = 1e-5;
cfg.callback = @plot_surface_grasp_points;
cfg.full_hessian = true;
cfg.surfaceImage = surfaceImage;
cfg.scale = scale;

d = size(x_init,1) / 2;

% set up zeros functions (since we don't need them)
A_eq = 0;
b_eq = 0;
A_ineq = [-eye(2*d); eye(2*d)];
b_ineq = [zeros(2*d,1); gridDim*ones(2*d,1)];
Q = zeros(2*d, 2*d);
q = zeros(1, 2*d);

f = @(x) (gp_cov(gpModel, x(1:d,1)') + nu*gp_cov(gpModel, x(d+1:2*d,1)'));
g = @(x) 0;
h = @(x) (surface_and_antipodality_functions(x, gpModel));

x_grasp = penalty_sqp(x_init, Q, q, f, A_ineq, b_ineq, A_eq, b_eq, g, h, cfg);    

end

