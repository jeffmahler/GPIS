function mu = gp_mean(gpModel, x, use_derivative)
%GP_MEAN predicts the mean value of the function at points x conditioned on
%   the data specified in the model |gpModel|
    if use_derivative
        Mx = linear_mean_derivative(gpModel.meanFunc, gpModel.hyp.mean, x);
        Kxxp = se_cov_derivative(gpModel.covFunc, gpModel.hyp.cov, gpModel.beta, gpModel.training_x, x);         
    else
        Mx = feval(gpModel.meanFunc{:}, gpModel.hyp.mean, x);
        Kxxp = feval(gpModel.covFunc{:}, gpModel.hyp.cov, gpModel.training_x, x);  
    end
    mu = Mx + Kxxp' * gpModel.alpha;
end

