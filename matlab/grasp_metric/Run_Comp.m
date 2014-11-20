
function [T_Q,contacts,HIST,varargout] = Run_Comp(experimentConfig,gpModel,shapeParams,img,constructionResults,grasp_set)
    cone_angle = atan(experimentConfig.frictionCoef);
    num_contacts = 2; 

    num_grasps = 100;
    grid_size = sqrt(size(shapeParams.all_points,1)); 
    numSamples = 3; 
    contacts = {};
    useNormal = true; 
    [shape_samples,pdfs] = sample_shapes(gpModel, shapeParams.gridDim, numSamples,1);
    
    for i =1:num_grasps
        close all; 
        
        [cp cp_mc] = get_random_grasp(shapeParams.gridDim);

        
        figure;                            
        errorbar(avg_q,var_q)
        toc
        T_Q(i,1) = mn_Q;
        T_Q(i,2) = v_Q;
        [loa_1,Norms,pc_1,pn_1] = Compute_Distributions(gpModel,shapeParams,cp(1:2,:),img);
        [loa_2,Norms,pc_2,pn_2] = Compute_Distributions(gpModel,shapeParams,cp(3:4,:),img);
        
        [c1_emps,n1_emps] = sample_loas(gpModel, loa_1, 5000,cp(1:2,:));
        [c2_emps,n2_emps] = sample_loas(gpModel, loa_2, 5000,cp(3:4,:));
%          
%       divg_contact = plot_mc_contact( loa_1,contact_emps,pc_1 )
%         
%       divg_normals = plot_mc_normals( loa_1,Norms,norm_emps,pn_1)
%         
       
        [avg_q1,var_q1] = sample_known_dist( loa_1,loa_2,pc_1,pc_2,pn_1,pn_2,Norms,numSamples,cone_angle,shapeParams.com,c1_emps,c2_emps,n1_emps,n2_emps);
        figure;
        errorbar(avg_q1,var_q1);
        ca  = atan(experimentConfig.frictionCoef);
        fc = experimentConfig.frictionCoef;
        [E_Q,lb] = compute_lb( loa_1,loa_2,Norms,pc_1,pn_1,pc_2,pn_2, shapeParams.com,ca,fc,gpModel,grasp_set);
        T_Q(i,3:4) = [E_Q,lb]; 
        contacts{i} = cp;
        HIST{i} = hst;
    end
    T_Q(:,5) = T_Q(:,3)/norm(T_Q(:,3)) - T_Q(:,4)/norm(T_Q(:,4));
end