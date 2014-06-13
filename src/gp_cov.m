function Sigma = gp_cov(gpModel, x)
%GP_COV predicts the covariance value of the function at points x conditioned on
%   the data specified in the model |gpModel|
    Kxx = feval(gpModel.covFunc{:}, gpModel.hyp.cov, x);    
    Kxxp = feval(gpModel.covFunc{:}, gpModel.hyp.cov, gpModel.training_x, x);
    v = solve_chol(gpModel.L, Kxxp) / gpModel.sl;
    Sigma = Kxx - Kxxp' * v;
    
    % force symmetric due to numerical issues
    [U, S, V] = svd(Sigma);
    Sigma = U * S * U';
end


