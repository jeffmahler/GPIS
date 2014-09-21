function hyperResults = ...
    run_hyperparam_search(shapeNames, gripScales, dataDir, outputDir,...
        experimentConfig, optimizationParams)
% RUN_HYPERPARAM_SEARCH Runs a grid search for the locally hyperparameters
% of the the antipodal grasp optimization.
% Hardcoded to take min, max, and increments for:
%   merit_coeff_increase_ratio - the rate at which the penalty increases
%   initial_penalty_coeff - the initial penalty coefficient
%   initial_trust_box - the initial size of the trust region
%   trust_shrink_ratio - the rate at which to shrink the trust region
%   trust_expand_ratio - the rate at which to shrexpandink the trust region
%   nu - the penalty on distance to the center of mass

numShapes = size(shapeNames, 2);
numContacts = 2;
coneAngle = atan(experimentConfig.frictionCoef);

optimizationParams.merit_coeff_increase_ratio = experimentConfig.min_mcir;
optimizationParams.initial_penalty_coeff = experimentConfig.min_ipc;
optimizationParams.initial_trust_box_size = experimentConfig.min_itbs;
optimizationParams.trust_shrink_ratio = experimentConfig.min_tsr;
optimizationParams.trust_expand_ratio = experimentConfig.min_ter;
optimizationParams.nu = experimentConfig.min_nu;

hyperResults = cell(numShapes,0);

% loop through shapes, calculate quality for each value of each
% hyperparameter
for i = 1:numShapes
    filename = shapeNames{i};
    [gpModel, shapeParams, shapeSamples, constructionResults] = ...
        load_experiment_object(filename, dataDir, optimizationParams.scale);
   
    % compute the width and scale
    dim = shapeParams.gridDim;
    gripScale = gripScales{i};
    gripWidth = gripScale * experimentConfig.objScale * dim;
    plateWidth = gripWidth * experimentConfig.plateScale;
    plateWidth = uint16(round(plateWidth));
    graspSigma = experimentConfig.graspSigma;
    
    optimizationParams.grip_width = gripWidth;
    
    predGrid = constructionResults.predGrid;
    surfaceImage = constructionResults.newSurfaceImage;
    numSamples = size(shapeSamples, 2);
    
    if experimentConfig.visOptimization
        optimizationParams.surfaceImage = surfaceImage;
    end
    
    % get initial grasps using com
    initGrasps = zeros(4, experimentConfig.numGrasps);
    for j = 1:experimentConfig.numGrasps
%        figure;
%        imshow(constructionResults.surfaceImage);
%         hold on;
        useNormalDirection = true;
        if rand(1) < 0.5
            useNormalDirection = false;
        end
        initGrasps(:,j) = ...
                get_initial_antipodal_grasp(constructionResults.predGrid, ...
                    useNormalDirection);
    end
    
    index = 1;
    optimizationParams.merit_coeff_increase_ratio = experimentConfig.min_mcir;    
    while optimizationParams.merit_coeff_increase_ratio <= experimentConfig.max_mcir
        optimizationParams.initial_penalty_coeff = experimentConfig.min_ipc;
        
        while optimizationParams.initial_penalty_coeff <= experimentConfig.max_ipc
            optimizationParams.initial_trust_box_size = experimentConfig.min_itbs;
            
            while optimizationParams.initial_trust_box_size <= experimentConfig.max_itbs
                optimizationParams.trust_shrink_ratio = experimentConfig.min_tsr;
                
                while optimizationParams.trust_shrink_ratio <= experimentConfig.max_tsr
                    optimizationParams.trust_expand_ratio = experimentConfig.min_ter;
                    
                    while optimizationParams.trust_expand_ratio <= experimentConfig.max_ter
                        optimizationParams.nu = experimentConfig.min_nu;
                    
                        while optimizationParams.nu <= experimentConfig.max_nu
                            fprintf('Evaluating quality of the following parameter choices on shape %s:\n', filename);
                            fprintf('merit_coeff_increase_ratio: %f\n', optimizationParams.merit_coeff_increase_ratio);
                            fprintf('initial_penalty_coef: %f\n', optimizationParams.initial_penalty_coeff);
                            fprintf('initial_trust_box_size: %f\n', optimizationParams.initial_trust_box_size);
                            fprintf('trust_shrink_ratio: %f\n', optimizationParams.trust_shrink_ratio);
                            fprintf('trust_expand_ratio: %f\n', optimizationParams.trust_expand_ratio);
                            fprintf('nu: %f\n', optimizationParams.nu);
                            
                            mn_q_vec = zeros(1, experimentConfig.numGrasps);
                            v_q_vec = zeros(1, experimentConfig.numGrasps);
                            p_fc_vec = zeros(1, experimentConfig.numGrasps);
                            opt_val_vec = zeros(1, experimentConfig.numGrasps);
                            opt_time_vec = zeros(1, experimentConfig.numGrasps);
                            x_grasp_vec = zeros(4, experimentConfig.numGrasps);
                            satisfied_vec = zeros(1, experimentConfig.numGrasps);
                            success_vec = zeros(1, experimentConfig.numGrasps);
                            %optimizationParams.surfaceImage = constructionResults.surfaceImage;
                            
                            for g = 1:experimentConfig.numGrasps
                                initGrasp = initGrasps(:,g);
                                
                                startTime = tic;
                                [optGrasp, optVal, allIters, constSatisfaction] = ...
                                    find_uc_mean_q_grasp_points(initGrasp, gpModel, ...
                                        optimizationParams, shapeParams.gridDim, shapeParams.com, ...
                                        predGrid, coneAngle, experimentConfig.numBadContacts, ...
                                        plateWidth, gripWidth, graspSigma);
                                optimizationTime = toc(startTime);
                                %opt_success = true;
                                %x_grasp = x_init;
                                x_grasp_vec(:,g) = optGrasp;
                                opt_time_vec(g) = optimizationTime;
                                satisfied_vec(g) = constSatisfaction;
                                opt_val_vec(g) = optVal;
                                
                                bestLoa = create_ap_loa(initGrasp, experimentConfig.gripWidth);
                                bestGraspSamples = sample_grasps(bestLoa, experimentConfig.graspSigma, numSamples);
                                [mn_q, v_q, success, p_fc] = mc_sample_fast(predGrid.points, ...
                                                                coneAngle, bestGraspSamples, numContacts, ...
                                                                shapeSamples, shapeParams.gridDim, ...
                                                                shapeParams.surfaceThresh, ...
                                                                experimentConfig.numBadContacts, ...
                                                                plateWidth, ...
                                                                experimentConfig.visSampling);
                                fprintf('Grasp %d mean q: %f var q: %f prob fc %f\n', g, mn_q, v_q, p_fc);
%                                 [mn_q2, v_q2, success] = MC_sample(gpModel, points, ...
%                                     coneAngle, cp, numContacts, shapeParams.com, ...
%                                     20, ...
%                                     shapeParams.surfaceThresh, 10);
                                mn_q_vec(g) = mn_q;
                                v_q_vec(g) = v_q;
                                p_fc_vec(g) = p_fc;
                                success_vec(g) = success;
                            end
            
                            % log results
                            paramResults = struct();
                            paramResults.cfg = optimizationParams;
                            paramResults.mn_q = mn_q_vec;
                            paramResults.max_mn_q = max(mn_q_vec);
                            paramResults.mean_mn_q = mean(mn_q_vec);
                            paramResults.v_q = v_q_vec;
                            paramResults.p_fc = p_fc_vec;
                            paramResults.opt_vals = opt_val_vec;
                            paramResults.opt_times = opt_time_vec;
                            paramResults.satisfied= satisfied_vec;
                            paramResults.opt_times = opt_time_vec;
                            paramResults.success = success_vec;
                            
                            % results are organized as shape, iteration
                            hyperResults{i, index} = paramResults;
                            
                            save(sprintf('%s/temp_hyp_results.mat', outputDir), 'hyperResults');

                            index = index+1;
                            optimizationParams.nu = optimizationParams.nu * experimentConfig.scale_nu;
                        end
                        optimizationParams.trust_expand_ratio = optimizationParams.trust_expand_ratio + experimentConfig.inc_ter;
                    end
                    optimizationParams.trust_shrink_ratio = optimizationParams.trust_shrink_ratio + experimentConfig.inc_tsr;
                end
                optimizationParams.initial_trust_box_size = optimizationParams.initial_trust_box_size + experimentConfig.inc_itbs;
            end
            optimizationParams.initial_penalty_coeff = optimizationParams.initial_penalty_coeff + experimentConfig.inc_ipc;
        end
        optimizationParams.merit_coeff_increase_ratio = optimizationParams.merit_coeff_increase_ratio + experimentConfig.inc_mcir;
    end
end

end

