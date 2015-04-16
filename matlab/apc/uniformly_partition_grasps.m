function grasp_bins = uniformly_partition_grasps(grasps, num_bins)

num_grasps = size(grasps, 2);
grasp_bins = cell(1, num_bins);
grasps_per_bin = floor(num_grasps / num_bins);

remaining_indices = 1:num_grasps;
num_remaining_indices = size(remaining_indices, 2);

for i = 1:num_bins
    if i < num_bins
        grasp_bins{i} = cell(1, grasps_per_bin);
        bin_ind = randsample(remaining_indices, grasps_per_bin);
    else
        grasp_bins{i} = cell(1, num_remaining_indices);
        bin_ind = randsample(remaining_indices, num_remaining_indices);
    end
    
    num_to_add = size(bin_ind, 2);
    for j = 1:num_to_add
        grasp_bins{i}{j} = grasps{bin_ind(j)};
    end
    
    remaining_indices = setdiff(remaining_indices, bin_ind);
    num_remaining_indices = size(remaining_indices, 2);
end

end


