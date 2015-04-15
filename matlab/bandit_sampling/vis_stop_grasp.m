function regret_results = ...
    vis_stop_grasp(experiment_results,config)
%COMPUTE_AVERAGE_REGRET Summary of this function goes here
%   Detailed explanation goes here

if nargin < 5
    eps = 1e-3;
end

num_shapes = config.num_shapes; 
num_methods = config.num_methods; 
regret_results = cell(num_methods, 1);

for i = 1:num_shapes
    top_grasp = {}; 
    Value = zeros(5,5); 
    count = 1;
    shape_result = experiment_results{i}; 
    
    best_random = shape_result.random.best_grasp; 
    best_kehoe = shape_result.kehoe.best_grasp; 
    best_bayes_ucb = shape_result.bayes_ucbs.best_grasp; 
    best_thompson = shape_result.thompson.best_grasp; 
    best_gittins98 = shape_result.gittins98.best_grasp; 
    [best_value,best_grasp] = max(shape_result.grasp_values(:,3)); 
    
    
    Value(5,:) = shape_result.grasp_values(best_grasp,:);
    top_grasp{5} = shape_result.grasp_samples{best_grasp}; 
    
    Value(1,:) = shape_result.grasp_values(best_random,:);
    top_grasp{1} = shape_result.grasp_samples{best_random}; 
    
    Value(2,:) = shape_result.grasp_values(best_kehoe,:);
    top_grasp{2} = shape_result.grasp_samples{best_kehoe}; 
    
%     Value(3,:) = shape_result.grasp_values(best_bayes_ucb,:);
%     top_grasp{3} = shape_result.grasp_samples{best_bayes_ucb}; 
    
    Value(3,:) = shape_result.grasp_values(best_thompson,:);
    top_grasp{3} = shape_result.grasp_samples{best_thompson}; 
    
    Value(4,:) = shape_result.grasp_values(best_gittins98,:);
    top_grasp{4} = shape_result.grasp_samples{best_gittins98}; 
    
    Value(1,:) = shape_result.grasp_values(best_random,:);
    top_grasp{1} = shape_result.grasp_samples{best_random}; 
 
    img = shape_result.construction_results.newSurfaceImage; 
    shapeParams = shape_result.construction_results.predGrid; 
    visualize_value( Value,shapeParams,top_grasp,img, i);
end

end


function [ ] = visualize_value( Value,shapeParams,grasp_samples,surface_image,ind)

N = size(grasp_samples, 2); 

Names = cell(5,1); 
Names{1} = 'Monte-Carlo';
Names{2} = 'Adaptive';
Names{3} = 'MAB-Thompson';
Names{4} = 'MAB-Gittins';
Names{5} = 'Best in Set';
figure(ind);

 for i=1:N
     cp = grasp_samples{i}.cp;
     visualize_grasp(cp,shapeParams, surface_image, 4, 9,i,N,Value(i,3),Names); 
     %plot_grasp_arrows( surface_image, cp(1,:)', cp(3,:)', -(cp(1,:)-cp(2,:))', -(cp(3,:)-cp(4,:))', 4,4,i,N,Value(i,3))
 end
tightfig; 

figure(100 + ind);
nom_grid = reshape(shapeParams.tsdf, [40, 40]);
nom_grid = high_res_tsdf(nom_grid, 5);
nom_grid = nom_grid > 0;
%nom_grid = imresize(nom_grid, 5);
imshow(nom_grid);

end

