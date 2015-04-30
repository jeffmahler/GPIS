function grasp_bins = space_partition_grasps(grasps, num_bins)

num_grasps = size(grasps, 2);
min_dim = realmax * ones(1, 5);
max_dim = realmin * ones(1, 5);

grasp_centers = zeros(3, num_grasps);
grasp_angles = zeros(2, num_grasps);
grasp_dirs = zeros(3, num_grasps);
grasp_bins = cell(1, num_bins);

for i = 1:num_grasps
    % compute grasp centers and angles
    g1_open = grasps{i}.g1_open;
    g2_open = grasps{i}.g2_open;
    g_center = (g1_open + g2_open) / 2;
    g_dir = (g2_open - g1_open);
    g_dir = g_dir / norm(g_dir);
    if g_dir(1) < 0
        g_dir = -g_dir;
    end
    grasp_dirs(:,i) = g_dir;

    % put in the grasp pose buffers
    grasp_centers(:,i) = g_center';
    [az, elev, ~] = sph2cart(g_dir(1), g_dir(2), g_dir(3));
    grasp_angles(1,i) = az;
    grasp_angles(2,i) = elev;
    
    % update maxima, minima
    min_dim(1:3) = min([min_dim(1:3); g_center]);
    max_dim(1:3) = max([max_dim(1:3); g_center]);
    min_dim(4:5) = min([min_dim(4:5); az, elev]);
    max_dim(4:5) = max([max_dim(4:5); az, elev]);
end

grasp_vecs = [grasp_centers; grasp_angles]';
Sig = cov(grasp_vecs);
Sig_sqrt = sqrtm(Sig);
grasp_vecs_whitened = inv(Sig_sqrt) * grasp_vecs';
cluster_ind = kmeans(grasp_vecs_whitened', num_bins, 'Distance', 'sqeuclidean', ...
    'Replicates', 5);

for i = 1:num_bins
    grasps_per_dim = sum(cluster_ind == i);
    grasp_ind = find(cluster_ind == i);
    grasp_bins{i} = cell(1, grasps_per_dim);
    for j = 1:size(grasp_ind, 1)
        grasp_bins{i}{j} = grasps{grasp_ind(j)};
    end
end

% compute bin increments
% delta_dims = (max_dim - min_dim) ./ bins_per_dim;
% for i = 1:5
%     cur_min = min_dim(i);
%     cur_max = min_dim(i) + delta_dims(i);
% 
%     while cur_min < max_dim(i)
%         
%         
% 
%         cur_min = cur_min(i) + delta_dims(i);
%         cur_max = cur_min(i) + delta_dims(i);
%     end
% 
% end

