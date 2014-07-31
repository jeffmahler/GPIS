
function [T_Q,contacts,HIST,varargout] = Run_Comp(experimentConfig,gpModel,shapeParams,img,constructionResults)
    cone_angle = atan(experimentConfig.frictionCoef);
    num_contacts = 2; 

    num_grasps = 1;
    grid_size = sqrt(size(shapeParams.all_points,1)); 
    numSamples = 1000; 
    contacts = {};
    useNormal = true; 
    [shape_samples,pdfs] = sample_shapes(gpModel, shapeParams.gridDim, numSamples);
    
    for i =1:num_grasps
        close all; 
        
        cp = get_random_grasp(shapeParams.gridDim);

        tic 
        [mn_Q,v_Q,success,p_fc,hst] = mc_sample_fast(shapeParams.all_points, cone_angle, cp, ...
                                        num_contacts, shape_samples, shapeParams.gridDim);
        
        toc
        T_Q(i,1) = mn_Q;
        T_Q(i,2) = v_Q;
        [loa_1,Norms,pc_1,pn_1] = Compute_Distributions(gpModel,shapeParams,cp(1:2,:),img);
        
        [contact_emps,norm_emps] = sample_loas(gpModel, loa_1, numSamples,cp(1:2,:));
        
        divg_contact = plot_mc_contact( loa_1,contact_emps,pc_1 )
        
        divg_normals = plot_mc_normals( loa_1,Norms,norm_emps,pn_1)
        
        [loa_2,Norms,pc_2,pn_2] = Compute_Distributions(gpModel,shapeParams,cp(3:4,:),img);

        ca  = atan(experimentConfig.frictionCoef);
        fc = experimentConfig.frictionCoef;
        [E_Q,lb] = compute_lb( loa_1,loa_2,Norms,pc_1,pn_1,pc_2,pn_2, shapeParams.com,ca,fc,gpModel);
        T_Q(i,3:4) = [E_Q,lb]; 
        contacts{i} = cp;
        HIST{i} = hst;
    end
    T_Q(:,5) = T_Q(:,3)/norm(T_Q(:,3)) - T_Q(:,4)/norm(T_Q(:,4));
end