function [expQGrasp, expPGrasp, avgSampleTime] = ...
    find_grasp_sampling(predGrid, experimentConfig, shapeParams, ...
                        shapeSamples, surfaceImage, scale, maxIters, ...
                        gripWidth, plateWidth, sampleGrasps)
%FIND_GRASP_SAMPLING Find the best grasp via sampling shapes
% The quality we are using is an estimate of the probability of force
% closure, P(Q_FC > 0)

if nargin < 8
   gripWidth = intmax; 
end
if nargin < 9
   plateWidth = 1; 
end
if nargin < 10
   sampleGrasps = true; 
end

attemptedGrasps = [];
fcQ = [];
fcV = [];
fcP = [];
maxQ = [];
maxP = [];
gs = [];
k = 0;

d = 2;
numContacts = 2;
coneAngle = atan(experimentConfig.frictionCoef);
length = experimentConfig.arrowLength;
numSamples = size(shapeSamples, 2);
avgSampleTime = 0;

while k < maxIters
    fprintf('%d\n', k);
    
    if rand() > 0.5
        useNormsForInit = true;
    else
        useNormsForInit = false;
    end
    
    % get random grasp
    randGrasp = get_initial_antipodal_grasp(predGrid, useNormsForInit);
    %randGrasp = [12; 12; 24; 12];
    
    % make sure our sample satisfies the width constraint
    if norm(randGrasp(1:d,1) - randGrasp(d+1:2*d,1)) > gripWidth
        continue;
    end
    
    loa = create_ap_loa(randGrasp, experimentConfig.gripWidth);
    
    if sampleGrasps
        randGraspSamples = sample_grasps(loa, experimentConfig.graspSigma, numSamples);
    else
        randGraspSamples = sample_grasps(loa, 0, numSamples);
    end
    
    % evaluate FC on mean shape
    startTime = tic;
    [mn_q, v_q, success, p_fc] = mc_sample_fast(predGrid.points, ...
                                    coneAngle, randGraspSamples, numContacts, ...
                                    shapeSamples, shapeParams.gridDim, ...
                                    shapeParams.surfaceThresh, ...
                                    experimentConfig.numBadContacts, ...
                                    plateWidth, ...
                                    experimentConfig.visSampling, false, ...
                                    experimentConfig.qScale);
    duration = toc(startTime);
                                
    % add quality to running count
    if success
        attemptedGrasps = [attemptedGrasps; randGrasp'];
        fcQ = [fcQ; mn_q];
        fcV = [fcV; v_q];
        fcP = [fcP; p_fc];
        
        maxQ = [maxQ; max(fcQ)];
        maxP = [maxP; max(fcP)];
        
        gs = [gs; randGraspSamples];
        
        avgSampleTime = avgSampleTime + duration;
        
        k = k+1;
        
        figure(23);
        visualize_grasp(randGrasp, predGrid, surfaceImage, scale, length, ...
            plateWidth, gripWidth);
        title('Best Grasp', 'FontSize', 15);
        fprintf('Q = %.03f\n', fcP);
    end
end

figure(24);
plot(maxQ, 'LineWidth', 2);
title('Best E[Q] vs Samples');
xlabel('# Samples');
ylabel('E[Q]');

figure(25);
plot(maxP, 'LineWidth', 2);
title('Best P(FC) vs Samples');
xlabel('# Samples');
ylabel('P(FC)');

% choose the grasp with maximum expected FC quality
bestQGraspIndices = find(fcQ == max(fcQ));
expQGrasp = struct();
expQGrasp.bestGrasp = attemptedGrasps(bestQGraspIndices(1), :);
expQGrasp.Q = fcQ(bestQGraspIndices(1));
expQGrasp.V = fcV(bestQGraspIndices(1));
expQGrasp.P = fcP(bestQGraspIndices(1));
expQGrasp.samples = gs(bestQGraspIndices(1), :);

% choose the grasp with maximum probabilty of force closure 
bestPGraspIndices = find(fcP == max(fcP));
expPGrasp = struct();
expPGrasp.bestGrasp = attemptedGrasps(bestPGraspIndices(1), :);
expPGrasp.Q = fcQ(bestPGraspIndices(1));
expPGrasp.V = fcV(bestPGraspIndices(1));
expPGrasp.P = fcP(bestPGraspIndices(1));
expPGrasp.samples = gs(bestPGraspIndices(1), :);

% calculate average sample time
avgSampleTime = avgSampleTime / maxIters;

% plot grasp
figure(23);
visualize_grasp(expQGrasp.bestGrasp', predGrid, surfaceImage, scale, length);
title('Best Grasp', 'FontSize', 15);
        
end

