function [grasp_samples] = collect_samples_grasps(gpModel, grasps, numSamples)
    if nargin < 3
       numSamples = 1500; 
    end
    
    num_grasps = size(grasps,1);
    grasp_samples = cell(num_grasps,1);
    cp = zeros(4,2);
    
    for i =1:num_grasps
        close all; 
        
        cp(1,:) = grasps(i, 1:2);
        cp(2,:) = grasps(i, 3:4); 
        cp(3,:) = cp(2,:);
        cp(4,:) = cp(1,:);
       % plot_grasp_arrows( constructionResults.newSurfaceImage, cp(1,:)', cp(3,:)', (cp(1,:)-cp(2,:))', (cp(3,:)-cp(4,:))', 2,2)
        loa_1 = compute_loa(cp(1:2,:));
        loa_2 = compute_loa(cp(3:4,:));
        
        [c1_emps,n1_emps] = sample_loas(gpModel, loa_1, numSamples, cp(1:2,:));
        [c2_emps,n2_emps] = sample_loas(gpModel, loa_2, numSamples, cp(3:4,:));
        
        grasp_samples{i} = struct(); 
        grasp_samples{i}.cp = cp; 
        grasp_samples{i}.c1_emps = c1_emps; 
        grasp_samples{i}.c2_emps = c2_emps; 
        
        grasp_samples{i}.n1_emps = n1_emps; 
        grasp_samples{i}.n2_emps = n2_emps; 
        
        grasp_samples{i}.loa_1 = loa_1; 
        grasp_samples{i}.loa_2 = loa_2;
        
        grasp_samples{i}.current_iter = 1; 
        grasp_samples{i}.q = [];
    end
   
end


