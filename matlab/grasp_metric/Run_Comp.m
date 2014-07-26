
function [T_Q] = Run_Comp(experimentConfig,gpModel,shapeParams,img)
    cone_angle = atan(experimentConfig.frictionCoef);
    num_contacts = 2; 

    num_grasps = 100;
    grid_size = sqrt(size(shapeParams.all_points,1)); 
    numSamples = 100; 
    
    [shape_samples,pdfs] = sample_shapes(gpModel, shapeParams.gridDim, numSamples);
    
    for i =1:num_grasps
        close all; 
        cp = get_random_grasp(grid_size);
%         cp1 = [12.5 1; 12.5 25];
%         cp2 = [12.5 25; 12.5 1];
% 
%         cp = [cp1; cp2];
        [loa_1,Norms,pc_1,pn_1] = Compute_Distributions(gpModel,shapeParams,cp(1:2,:),img);

        [loa_2,Norms,pc_2,pn_2] = Compute_Distributions(gpModel,shapeParams,cp(3:4,:),img);
        tic 
        [mn_Q,v_Q,success] = mc_sample_fast(shapeParams.all_points, cone_angle, cp, ...
                                        num_contacts, shape_samples, shapeParams.gridDim);
        toc
        T_Q(i,1) = mn_Q;
        [loa_1,Norms,pc_1,pn_1] = Compute_Distributions(gpModel,shapeParams,cp(1:2,:),img);

        [loa_2,Norms,pc_2,pn_2] = Compute_Distributions(gpModel,shapeParams,cp(3:4,:),img);

        ca  = atan(experimentConfig.frictionCoef);
        fc = experimentConfig.frictionCoef;
        [E_Q,lb] = compute_lb( loa_1,loa_2,Norms,pc_1,pn_1,pc_2,pn_2, shapeParams.com,ca,fc,gpModel);
        T_Q(i,2:3) = [E_Q,lb]; 
    end
end