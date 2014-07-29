function val = uc_mean_q_penalty(x, gpModel, nu, ...
    coneAngle, shapeParams, badContactThresh, gamma)
   
    if nargin < 7
       gamma = 1.0;
    end


    % full covariance penaty if not specified
    d = size(x,1) / 2;
    Sig = gp_cov(gpModel, [x(1:d,1)'; x(d+1:2*d,1)'], [], true);
    
    numContacts = 2;
    visSampling = false;
    visHistogram = false;
    loaScale = 1.75;
    
    loa = create_ap_loa(x, loaScale);
    
    meanSample = {shapeParams};
    [mn_q, v_q, success] = mc_sample_fast(shapeParams.points, ...
                                    coneAngle, loa, numContacts, ...
                                    meanSample, shapeParams.gridDim, ...
                                    shapeParams.surfaceThresh, ...
                                    badContactThresh, ...
                                    visSampling, visHistogram);

    val = gamma*sum(diag(Sig)) - nu*mn_q;
end

