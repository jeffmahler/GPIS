function val = antipodality_penalty(x, gpModel, nu, lambda, com, gamma)
   
    % full covariance penaty if not specified
    if nargin < 6
       gamma = 1.0 
    end

    d = size(x,1) / 2;
    Sig = gp_cov(gpModel, [x(1:d,1)'; x(d+1:2*d,1)'], [], true);
    
    xp = [x(1:d,1)'; x(d+1:2*d,1)'];
    diff = xp(1,:)' - xp(2,:)';
    n = diff / norm(diff);
    v = xp(2,:)' - com';
    com_dist = norm(v - (v'*n) * n);
    
    val = gamma * sum(diag(Sig)) + nu*com_dist + lambda*norm(diff);
end

