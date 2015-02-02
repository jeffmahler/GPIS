function [ Grasps ] = Collect_Samples( experimentConfig,gpModel,shapeParams,img,constructionResults)
%COLLECT_SAMPLES Summary of this function goes here
%   Detailed explanation goes here
    cone_angle = atan(experimentConfig.frictionCoef);
    num_contacts = 2; 

    num_grasps = 100;
    grid_size = sqrt(size(shapeParams.all_points,1)); 
    numSamples = 1000; 
    contacts = {};
    useNormal = true; 
    [shape_samples,pdfs] = sample_shapes(gpModel, shapeParams.gridDim, numSamples);
    Grasps = [];
    for i=1:num_grasps
        cp = get_random_grasp(shapeParams.gridDim);
        
        [mn_Q,v_Q,success,p_fc,hst,contact_emps,normal_emps, coms] = mc_sample_fast(shapeParams.all_points, cone_angle, cp, ...
                                        num_contacts, shape_samples, shapeParams.gridDim);
        fric = zeros(size(hst))+experimentConfig.frictionCoef;
        
        moment_arm1 = contact_emps(:,1:2) - coms; 
        moment_arm2 = contact_emps(:,3:4) - coms; 
        
        Grasps = [Grasps; hst moment_arm1 moment_arm2 normal_emps];
    end



end

