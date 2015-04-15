function val = uc_mean_q_penalty(x, gpModel, nu, ...
    coneAngle, shapeParams, badContactThresh, plateWidth,...
    gripWidth, graspSigma)

    % full covariance penaty if not specified
    d = size(x,1) / 2;
    %a = [x(1:d,1)'; x(d+1:2*d,1)'];
    Sig = gp_cov(gpModel, [x(1:d,1)'; x(d+1:2*d,1)'], [], true);
    
    numContacts = 2;
    visSampling = false;
    visHistogram = false;
    
    loa = create_ap_loa(x, gripWidth);
    tangentDir = [-loa(1,2), loa(1,1)];
    tangentDir = tangentDir / norm(tangentDir);
    
    % sample grasps from sigma points
    sp1 = loa(1,:) + graspSigma * tangentDir;
    sp2 = loa(1,:) - graspSigma * tangentDir;
    sp3 = loa(2,:) + graspSigma * tangentDir;
    sp4 = loa(2,:) - graspSigma * tangentDir;
    spLoas = { loa, ...
               [sp1; loa(2,:); loa(2,:); sp1], ...
               [sp2; loa(2,:); loa(2,:); sp2], ...
               [loa(1,:); sp3; sp3; loa(1,:)], ...
               [loa(1,:); sp4; sp4; loa(1,:)], ...
               [sp1; sp3; sp3; sp1], ...
               [sp1; sp4; sp4; sp1], ...
               [sp2; sp3; sp3; sp2], ...
               [sp2; sp4; sp4; sp2]};
    loaWeights = [1, exp(-0.5), exp(-0.5), exp(-0.5), exp(-0.5), ...
                  exp(-1), exp(-1), exp(-1), exp(-1)];
    
    
    meanSample = {shapeParams};
    q_estimate = 0;
    p_estimate = 0;
    w_sum = 0;
    
    for i = 1:size(spLoas,2)
        graspSample = {spLoas{i}};

        [mn_q, v_q, success, p_fc] = mc_sample_fast(shapeParams.points, ...
                                        coneAngle, graspSample, numContacts, ...
                                        meanSample, shapeParams.gridDim, ...
                                        shapeParams.surfaceThresh, ...
                                        badContactThresh, plateWidth, gripWidth,...
                                        visSampling, visHistogram);
        q_estimate = q_estimate + loaWeights(i) * mn_q;
        p_estimate = p_estimate + loaWeights(i) * p_fc;
        w_sum = w_sum + loaWeights(i);
    end
    mn_q = q_estimate / w_sum;
    p_fc = p_estimate / w_sum; 
    
    % get uncertainty at contacts
%     loa = create_ap_loa(x, gripWidth);
%     fakeCom = [0,0];
%     vis = false;
%     [xp, ~, ~ ] = ...
%         find_contact_points(loa, numContacts, shapeParams.points, ...
%             shapeParams.tsdf, shapeParams.normals, ...
%             fakeCom, shapeParams.surfaceThresh, vis, plateWidth);
%     xp = xp';
%     Sig = gp_cov(gpModel, xp, [], false);
    
    % use probability of force closure
    val = nu*sum(diag(Sig)) - p_fc;
end

