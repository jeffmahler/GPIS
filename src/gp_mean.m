function mu = gp_mean(gpModel, x)
%GP_MEAN predicts the mean value of the function at points x conditioned on
%   the data specified in the model |gpModel|
    Mx = feval(gpModel.meanFunc{:}, gpModel.hyp.mean, x);
    Kxxp = feval(gpModel.covFunc{:}, gpModel.hyp.cov, gpModel.training_x, x);
    mu = Mx + Kxxp' * gpModel.alpha;
end

