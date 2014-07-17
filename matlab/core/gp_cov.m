function Sigma = gp_cov(gpModel, x, Kxxp, use_derivative)
%GP_COV predicts the covariance value of the function at points x conditioned on
%   the data specified in the model |gpModel|

    if use_derivative
        Kxx = se_cov_derivative(gpModel.covFunc, gpModel.hyp.cov, 0, x);    
        if size(Kxxp,1) == 0
            Kxxp = se_cov_derivative(gpModel.covFunc, gpModel.hyp.cov, gpModel.beta, gpModel.training_x, x);
        end
    else
        Kxx = feval(gpModel.covFunc{:}, gpModel.hyp.cov, x);    
        if size(Kxxp,1) == 0
            Kxxp = feval(gpModel.covFunc{:}, gpModel.hyp.cov, gpModel.training_x, x);
        end
    end
    v = solve_chol(gpModel.L, Kxxp) / gpModel.sl;
    Sigma = Kxx - Kxxp' * v;
    
    % force symmetric due to numerical issues
    Sigma = (Sigma + Sigma') / 2;
end


