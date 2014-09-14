function val = uc_mean_q_penalty(x, gpModel, nu, ...
    coneAngle, shapeParams, badContactThresh, plateWidth)

    % full covariance penaty if not specified
    d = size(x,1) / 2;
    Sig = gp_cov(gpModel, [x(1:d,1)'; x(d+1:2*d,1)'], [], true);
    
    numContacts = 2;
    visSampling = false;
    visHistogram = false;
    loaScale = 1.75;
    
    loa = create_ap_loa(x, loaScale);
%     tangentDir = [-loa(2,1); loa(1,1)];
%     tangentDir = tangentDir / norm(tangentDir);

    
    meanSample = {shapeParams};
    graspSample = {loa};
    [mn_q, v_q, success] = mc_sample_fast(shapeParams.points, ...
                                    coneAngle, graspSample, numContacts, ...
                                    meanSample, shapeParams.gridDim, ...
                                    shapeParams.surfaceThresh, ...
                                    badContactThresh, plateWidth, ...
                                    visSampling, visHistogram);

    val = nu*sum(diag(Sig)) - mn_q;
end

