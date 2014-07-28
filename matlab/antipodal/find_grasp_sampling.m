function [bestGrasp, bestQ, bestV] = ...
    find_grasp_sampling(predGrid, experimentConfig, shapeParams, ...
                        shapeSamples, surfaceImage, scale, maxIters)
%FIND_GRASP_SAMPLING Summary of this function goes here
%   Detailed explanation goes here

attemptedGrasps = [];
fcQ = [];
fcV = [];
k = 0;

numContacts = 2;
coneAngle = atan(experimentConfig.frictionCoef);
length = experimentConfig.arrowLength;
useNormsForInit = true;

while k < maxIters
    fprintf('%d\n', k);
    % get random grasp
    randGrasp = get_initial_antipodal_grasp(predGrid, useNormsForInit);
    %randGrasp = [6; 4; 15; 2];
    loa = create_ap_loa(randGrasp, experimentConfig.loaScale);

    % evaluate FC on mean shape
    [mn_q, v_q, success] = mc_sample_fast(predGrid.points, ...
                                coneAngle, loa, numContacts, ...
                                shapeSamples, shapeParams.gridDim, ...
                                shapeParams.surfaceThresh, ...
                                experimentConfig.numBadContacts, ...
                                experimentConfig.visSampling, false);

    % add quality to running count
    if success
        attemptedGrasps = [attemptedGrasps; randGrasp'];
        fcQ = [fcQ; mn_q];
        fcV = [fcV; v_q];
        k = k+1;
        
        figure(23);
        visualize_grasp(randGrasp, predGrid, surfaceImage, scale, length);
        title('Best Grasp', 'FontSize', 15);
        fprintf('Q = %f\n', fcQ);
    end
end

% choose the grasp with maximum FC quality and evaluate
bestGraspIndices = find(fcQ == max(fcQ));
bestGrasp = attemptedGrasps(bestGraspIndices(1), :);
bestQ = fcQ(bestGraspIndices(1));
bestV = fcV(bestGraspIndices(1));

% plot grasp
figure(23);
visualize_grasp(bestGrasp', predGrid, surfaceImage, scale, length);
title('Best Grasp', 'FontSize', 15);
        
end

