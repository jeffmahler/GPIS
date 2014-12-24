function [grasp_samples,best_grasp,Value] = collect_samples(experimentConfig,num_grasps,gpModel,shapeParams,fc_iter)
 
    num_contacts = 2; 
    numSamples = experimentConfig.num_grasp_samples; 
   
    
    grid_size = sqrt(size(shapeParams.all_points,1)); 
    
    grasp_samples = {};
    useNormal = true; 
    if(nargin < 5)
        fc_iter = 1; 
    end
    
   
    
    pose_samples  = sample_pose_2d(numSamples,experimentConfig.pose_var); 
    shape_samples = sample_shapes_pose(gpModel, grid_size, numSamples,true,2,pose_samples);
    dim = shape_samples{1}.dim; 
    Value = zeros(num_grasps,5); 
    
    parfor i =1:num_grasps
  
        grasp_samples{i} = struct(); 
        [cp cp_mc] = get_random_grasp(dim,dim,1);
        c1_emps = zeros(numSamples,2); 
        c2_emps = zeros(numSamples,2); 
        Q_samps = zeros(numSamples,1); 
        n1_emps = zeros(numSamples,2); 
        n2_emps = zeros(numSamples,2); 
        for k =1:numSamples
            shapeSample = shape_samples{k};
            allTsdf = shapeSample.tsdf;
            allNorm = shapeSample.normals;
            allPoints = shapeSample.all_points; 
            [contacts, normal, bad]= find_contact_points(cp, 2, allPoints, allTsdf, allNorm);
            com = 0.5*shapeParams.com; 
            fc = experimentConfig.friction_coef + experimentConfig.fric_var*rand();
            Q = evaluate_grasp_lite(normal(:,1),normal(:,2),contacts(:,1),contacts(:,2),com,fc);
            
            c1_emps(k,:) = contacts(:,1)';  
            Q_samps(k,1) = Q; 
            count_s = sum(Q_samps); 
            Value(i,:) = [count_s, k, (count_s+1)/(k+2), 0, 0];  
           
            
            
            c2_emps(k,:) = contacts(:,2)'; 
            
            n1_emps(k,:) = normal(:,1)'; 
            n2_emps(k,:) = normal(:,2)'; 
            
            
        end
        
        
        loa_1 = compute_loa(cp(1:2,:));
        loa_2 = compute_loa(cp(3:4,:));
     
        
        grasp_samples{i} = struct(); 
        grasp_samples{i}.cp = cp; 
        grasp_samples{i}.c1_emps = c1_emps; 
        grasp_samples{i}.c2_emps = c2_emps; 
        grasp_samples{i}.Q_samps = Q_samps; 
        grasp_samples{i}.fc = experimentConfig.friction_coef + fc_iter*0.04*rand(1,numSamples);
        
        com = zeros(numSamples,2); 
        com(:,1) = 3*randn(numSamples,1)+shapeParams.com(1); 
        com(:,2) = 3*randn(numSamples,1)+shapeParams.com(2); 
        
        grasp_samples{i}.com = com*0.5; 
        grasp_samples{i}.n1_emps = n1_emps; 
        grasp_samples{i}.n2_emps = n2_emps; 
        
        grasp_samples{i}.loa_1 = loa_1; 
        grasp_samples{i}.loa_2 = loa_2;
        
        grasp_samples{i}.current_iter = 1; 
        grasp_samples{i}.num_samples = numSamples; 
        grasp_samples{i}.q = [];
      
    end
    save('marker_bandit_values_pfc', 'Value');
    [best_grasp,v] = max(Value(:,3)); 
   
end

