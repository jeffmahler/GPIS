function [x, x_all_iters, success] = penalty_sqp(x0, Q, q, f, A_ineq, b_ineq, A_eq, b_eq, g, h, user_cfg)
%PENALTY_SQP    solves constrained optimization problem
%
%   minimize (1/2) x'*Q*x + x'*q + f(x)
%   subject to
%       A_ineq*x <= b_ineq
%       A_eq*x == b_eq
%       g(x) <= 0
%       h(x) == 0
%
%   x0: initialization
%
%   f, g, h are function handles.
%   These functions will be numerically differentiated unless you specify otherwise.
%   To use an analytic gradient and hessian for f, set
%   cfg.f_use_numerical = false
%   Then f should have the form
%   [y, grad, hess] = f(x)
%   Similarly, g and h, set cfg.g_use_numerical and cfg.h_use_numerical and provide
%   [y, grad] = g(x)
%   


assert(size(x0,2) == 1);

cfg = {};
cfg.improve_ratio_threshold = .25;
cfg.min_trust_box_size = 1e-4;
cfg.min_approx_improve = 1e-4;
cfg.max_iter = 50;
cfg.trust_shrink_ratio = .1;
cfg.trust_expand_ratio = 1.5;
cfg.cnt_tolerance = 1e-4;
cfg.ineq_tolerance = 1e-4;
cfg.eq_tolerance = 1e-4;
cfg.prog_tolerance = 0;
cfg.max_merit_coeff_increases = 5;
cfg.merit_coeff_increase_ratio = 8;
cfg.initial_trust_box_size = 1;
cfg.initial_penalty_coeff = 1.;
cfg.max_penalty_iter = 5;
cfg.f_use_numerical = true;
cfg.g_use_numerical = true;
cfg.h_use_numerical = true;
cfg.full_hessian = true;
cfg.surfaceImage = zeros(1,1);
cfg.scale = 1;
cfg.lambda = 1;
cfg.nu = 1;
cfg.beta = 1;
cfg.fric_coef = 0;
cfg.min_init_dist = 3;
cfg.com_tol = 0;
cfg.grip_width = intmax;
cfg.plate_width = 1;
cfg.arrow_length = 1;
cfg.com = [0;0];
cfg.callback = [];

cfg = load_user_cfg(cfg, user_cfg);

disp('Optimizer parameters:');
disp(cfg)

% correct the empty matrices
if size(A_ineq,1) == 0
   A_ineq = zeros(1,size(x0,1));
   b_ineq = zeros(1,1);
end
if size(A_eq,1) == 0
   A_eq = zeros(1,size(x0,1));
   b_eq = zeros(1,1);
end

% First we find a point that satisfies linear constraints.
% We'll enforce these constraints exactly in the optimization that follows.
% (Whereas the nonlinear constraints will be initially violated and treated
% with penalties.)
[x,success] = find_closest_feasible_point(x0, A_ineq, b_ineq, A_eq, b_eq); 
if (~success)
    return;
end

trust_box_size = cfg.initial_trust_box_size; % The trust region will be a box around the current iterate x.
penalty_coeff = cfg.initial_penalty_coeff; % Coefficient of l1 penalties 
penalty_iter = 0;

if ~isempty(cfg.callback), cfg.callback(); end;

% TODO: Write the outer loop of the sqp algorithm, which repeatedly minimizes
% the merit function f(x) + penalty_coeff*( pospart(g(x)) + abs(h(x)) )
% Call minimize_merit_function defined below

% After this call, check to see if the
% constraints are satisfied.
% - If some constraint is violated, increase penalty_coeff by a factor of cfg.merit_coeff_increase_ratio
% You should also reset the trust region size to be larger than cfg.min_trust_box_size,
% which is used in the termination condition for the inner loop.
% - If all constraints are satisfied (which in code means if they are satisfied up to tolerance cfg.cnt_tolerance), we're done.
%
x_all_iters = x;
while penalty_iter <= cfg.max_penalty_iter    
    
    [x, trust_box_size, success] = minimize_merit_function(x, Q, q, ...
    f, A_ineq, b_ineq, A_eq, b_eq, g, h, cfg, penalty_coeff, trust_box_size);
	
    x_all_iters = [x_all_iters, x];
    num_iters = size(x_all_iters, 2);
    
    % reset trust box if necessary
    if trust_box_size < cfg.min_trust_box_size
        trust_box_size = cfg.initial_trust_box_size;
    end
    
    if (max(A_ineq*x - b_ineq) > cfg.cnt_tolerance) ...
            || (max(abs(A_eq*x - b_eq)) > cfg.cnt_tolerance) ...
            || (sum(g(x) > cfg.ineq_tolerance) > 0) ...
            || (sum(abs(h(x)) > cfg.eq_tolerance) > 0)% || ...
          %  norm(x - x_all_iters(:, num_iters-1)) > cfg.prog_tolerance) % there's at least 1 bad constraint
        max(A_ineq*x - b_ineq)
        max(abs(A_eq*x - b_eq))
        sum(g(x) > cfg.ineq_tolerance)
        sum(abs(h(x)) > cfg.eq_tolerance)
        
        fprintf('Penalty: %f\n', penalty_coeff);
        
        penalty_coeff = cfg.merit_coeff_increase_ratio * penalty_coeff;
    else
        break; % we're close enough
    end
    penalty_iter = penalty_iter + 1;
end

if penalty_iter > cfg.max_penalty_iter
    fprintf('Failed to find a solution that satisfies all of the constraints\n');
    success = false;
end
end

function full_cfg = load_user_cfg(full_cfg, user_cfg)
userkeys = fieldnames(user_cfg);
for iuserkey = 1:numel(userkeys)
    userkey = userkeys{iuserkey};
    if isfield(full_cfg, userkey)
        full_cfg.(userkey) = user_cfg.(userkey);
    else
        error(['Unknown parameter: ' userkey])
    end
end
end

function [x,success] = find_closest_feasible_point(x0, A_ineq, b_ineq, A_eq, b_eq)
% Find a point that satisfies linear constraints, if x0 doesn't
% satisfy them

success = true;
if any(A_ineq*x0 > b_ineq) || any(A_eq*x0 ~= b_eq)
    fprintf('initialization doesn''t satisfy linear constraints. finding the closest feasible point\n');

    cvx_begin
    cvx_quiet('true')
    variables('x(length(x0))')
    minimize('sum((x-x0).^2)')
    subject to
        A_ineq*x <= b_ineq;
        A_eq*x == b_eq;
    cvx_end
    
    if strcmp(cvx_status,'Failed')
        success = false;
        fprintf('Couldn''t find a point satisfying linear constraints\n')
        return;
    end
else
    x = x0;
end
end

function [x, trust_box_size, success] = minimize_merit_function(x, Q, q, ...
    f, A_ineq, b_ineq, A_eq, b_eq, g, h, cfg, penalty_coeff, trust_box_size)
    
    dim_x = length(x);


    success = true;
    sqp_iter = 1;

    fquadlin = @(x) q*x + .5*x'*(Q*x);
    hinge = @(x) sum(max(x,0));
    abssum = @(x) sum(abs(x));
    
    while  true
        % In this loop, we repeatedly construct a quadratic approximation
        % to the nonlinear part of the objective f and a linear approximation to the nonlinear
        % constraints f and g.
        fprintf('  sqp iter: %i\n', sqp_iter);
                    
        if cfg.f_use_numerical
            fval = f(x);
            fgrad = numerical_jac(f,x);
            %[fgrad, fhess] = numerical_grad_hess(f,x,cfg.full_hessian);
            % diagonal adjustment
%             mineig = min(eigs(fhess));
%             if mineig < 0
%                 fprintf('    negative hessian detected. adjusting by %.3g\n',-mineig);
%                 fhess = fhess + eye(dim_x) * ( - mineig);
%             end
        else
            [fval, fgrad, fhess] = f(x);
        end
        if cfg.g_use_numerical
            gval = g(x);
            gjac = numerical_jac(g,x);
        else           
            [gval, gjac] = g(x);
        end
        if cfg.h_use_numerical
            hval = h(x);
            hjac = numerical_jac(h,x);
        else
            [hval, hjac] = h(x);
        end
        fval
        gradquadlin = numeric_gradient(fquadlin, x);
        merit = fval + fquadlin(x) + penalty_coeff * ( hinge(gval) + abssum(hval) );
        
        while true 
            % This is the trust region loop
            % Using the approximations computed above, this loop shrinks
            % the trust region until the progress on the approximate merit
            % function is a sufficiently large fraction of the progress on
            % the exact merit function.
            
            fprintf('    trust region size: %.3g\n', trust_box_size);

            
            % YOUR CODE INSIDE CVX_BEGIN and CVX_END BELOW
			% Write CVX code to minimize the convex approximation to
            % the merit function, using the jacobians computed above.
            % It should create variable xp, which is the candidate for
            % updating x -> xp.
            % You should enforce the linear constraints exactly.
			% Make sure to include the constant term f(x) in the merit function
			% objective as the resulting cvx_optval is used further below.
            
            cvx_begin quiet
                variable xp(dim_x, 1);

				minimize( fquadlin(x) + f(x) + gradquadlin'*(xp - x) + fgrad*(xp - x) + ...
                    penalty_coeff * hinge(gval + gjac*(xp - x)) + ...
                    penalty_coeff * abssum(hval + hjac*(xp - x)) )
                subject to
                    norm(x - xp) <= trust_box_size;
                    A_ineq*xp <= b_ineq;
                    A_eq*xp == b_eq;
				
            cvx_end
            
            if strcmp(cvx_status,'Failed')
                fprintf('Failed to solve QP subproblem.\n');
                success = false;
                return;
            end
            
            
            model_merit = cvx_optval;
            new_merit = f(xp) + fquadlin(xp) + penalty_coeff * ( hinge(g(xp)) + abssum(h(xp)) ) ;
            approx_merit_improve = merit - model_merit;
            exact_merit_improve = merit - new_merit;
            merit_improve_ratio = exact_merit_improve / approx_merit_improve;
                        
            info = struct();
            info.trust_box_size = trust_box_size;
            info.cfg = cfg;
                      
            fprintf('      approx improve: %.3g. exact improve: %.3g. ratio: %.3g\n', approx_merit_improve, exact_merit_improve, merit_improve_ratio);
            if approx_merit_improve < -1e-5
                fprintf('Approximate merit function got worse (%.3e).\n',approx_merit_improve);
                fprintf('Either convexification is wrong to zeroth order, or you''re in numerical trouble\n');
                success = false;
                return;
            elseif approx_merit_improve < cfg.min_approx_improve
                fprintf('Converged: y tolerance\n');
                x = xp;
                if ~isempty(cfg.callback), cfg.callback(x,info); end
                return;
            elseif (exact_merit_improve < 0) || (merit_improve_ratio < cfg.improve_ratio_threshold)
                trust_box_size = trust_box_size * cfg.trust_shrink_ratio;
            else
                trust_box_size = trust_box_size * cfg.trust_expand_ratio;
                x = xp;
                if ~isempty(cfg.callback), cfg.callback(x,info); end
                break; % from trust region loop
            end
            
            if trust_box_size < cfg.min_trust_box_size
                fprintf('Converged: x tolerance\n');
                return;
            end
        end % tr
        sqp_iter = sqp_iter + 1;
    end % sqp

end



