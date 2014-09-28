function fval = soft_constraint_energy(x, gpModel, ...
    cfg, gridDim, com, shapeParams, ...
    coneAngle, badContactThresh, plateWidth, ...
    gripWidth, graspSigma)
% Treat the normal constraints as soft and optimize w/o gradients

% get dim
nu = cfg.nu;

% get objective value
p_fc_approx = uc_mean_q_penalty(x, gpModel, nu, coneAngle, shapeParams, ...
    badContactThresh, plateWidth, gripWidth, graspSigma);

% check constraints
fc_constraint = [friction_cone_constraint(x, gpModel, cfg.fric_coef, com, cfg.com_tol); ...
                    gripper_width_constraint(x, gpModel, cfg.grip_width)];
ap_constraint = surface_and_antipodality_functions(x, gpModel);

% approximate w/ infinity when constraints violated
g_penalty = fc_constraint; %realmax * (fc_constraint <= cfg.ineq_tolerance);
h_penalty = 5*abs(ap_constraint);%realmax * (abs(ap_constraint) <= cfg.eq_tolerance);

fval = p_fc_approx + sum(g_penalty) + sum(h_penalty);

end

