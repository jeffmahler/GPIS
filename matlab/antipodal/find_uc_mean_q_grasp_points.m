function [x_grasp, opt_val, x_all_iters, success] = ...
    find_uc_mean_q_grasp_points(x_init, gpModel, ...
    cfg, gridDim, com, shapeParams, ...
    coneAngle, badContactThresh, plateWidth, ...
    gripWidth, graspSigma, ...
    useUncertainty, forceAntipodal)
%FIND_ANTIPODAL_GRASP_POINTS Finds an antipodal set of grasp points

use_com = true;
if nargin < 5
    use_com = false;
end
if nargin < 12
    useUncertainty = true;
end
if nargin < 13
    forceAntipodal = true;
end

nu = cfg.nu;
if ~useUncertainty
   nu = 0; 
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

f = @(x) (uc_mean_q_penalty(x, gpModel, nu, coneAngle, shapeParams, ...
    badContactThresh, plateWidth, gripWidth, graspSigma));

cfg.com = shapeParams.com;
cfg.plate_width = plateWidth;

if forceAntipodal
    if use_com
        g = @(x) ([friction_cone_constraint(x, gpModel, cfg.fric_coef, shapeParams, ...
                                            gripWidth, plateWidth, com, cfg.com_tol); ...
                   gripper_width_constraint(x, gpModel, cfg.grip_width) ]);
        h = @(x) (surface_and_antipodality_functions(x, gpModel, shapeParams, ...
                  gripWidth, plateWidth));%, com));
    else
        g = @(x) ([friction_cone_constraint(x, gpModel, cfg.fric_coef, shapeParams, ...
                                            gripWidth, plateWidth) ; ...
                   gripper_width_constraint(x, gpModel, cfg.grip_width) ]);
        h = @(x) (surface_and_antipodality_functions(x, gpModel, shapeParams, ...
                  gripWidth, plateWidth));
    end
else
    % unconstrained
    g = @(x) ([0;0;0 ; ...
               gripper_width_constraint(x, gpModel, cfg.grip_width) ]);
    surfaceOnly = true;
    zeroCom = [0,0];
    if use_com
        h = @(x) (surface_and_antipodality_functions(x, gpModel, shapeParams, ...
                  gripWidth, plateWidth, com, surfaceOnly));
    else
        h = @(x) (surface_and_antipodality_functions(x, gpModel, shapeParams, ...
                  gripWidth, plateWidth, zeroCom, surfaceOnly));
    end
end

init = h(x_init);
[x_grasp, x_all_iters, success] = penalty_sqp(x_init, Q, q, f, A_ineq, b_ineq, A_eq, b_eq, g, h, cfg);    
opt_val = f(x_grasp);

end


