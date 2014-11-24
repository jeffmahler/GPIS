% Creates a 2D tsdf library given an initial directory

% define variables
data_dir = 'data/brown_dataset';
model_dir = 'data/grasp_transfer_models/brown_dataset';
downsample = 4;

% load models
filenames_filename = sprintf('%s/filenames.mat', model_dir);
tsdf_filename = sprintf('%s/tsdf_vectors_%d.mat', model_dir, downsample);
kd_tree_filename = sprintf('%s/kd_tree_%d.mat', model_dir, downsample);

S = load(filenames_filename);
filenames = S.filenames;

S = load(tsdf_filename);
tsdf_vectors = S.X;

S = load(kd_tree_filename);
kd_tree = S.kd_tree;

num_training = size(tsdf_vectors, 1);
num_points = size(tsdf_vectors, 2);
grid_dim = sqrt(num_points);

%% loop through models, generating and ranking a grasp set for each
debug = true;
scale = 2;
arrow_length = 10;
plate_width = 3;
grip_width = grid_dim;
bad_contact_thresh = 10;
num_contacts = 2;
friction_coef = 0.75;
surface_thresh = 0.25;
cone_angle = atan(friction_coef);

num_grasps = 500;
num_grasps_keep = 25;
grasps_scratch = zeros(num_grasps, 4);
grasp_q_scratch = zeros(num_grasps, 1);
grasps = zeros(num_training, num_grasps_keep, 4);
grasp_qualities = zeros(num_training, num_grasps_keep);

for i = 1:num_training
    filename = filenames{i};
    if mod(i, 10) == 0
       fprintf('Evaluating grasps on shape file %d: %s\n', i, filename);
    end
    tsdf = tsdf_vectors(i,:);
    tsdf = reshape(tsdf, [grid_dim, grid_dim]);
    
    if debug
        figure;
        imshow(tsdf);
        pause(2);
    end
    
    shape_image = imresize(tsdf, scale);
    shape_params = tsdf_to_shape_params(tsdf);
    shape_params.surfaceThresh = surface_thresh;
    shape_samples = {shape_params};
    
    for j = 1:num_grasps
        % randomly select a grasp
        r = rand();
        if r < 0.33
            g1 = get_initial_antipodal_grasp(shape_params, true);
        elseif r < 0.67
            g1 = get_initial_antipodal_grasp(shape_params, false);
        else
            [g1, g2] = get_random_grasp(grid_dim, grid_dim);
            g1 = [g1(1,:) g1(2,:)]';
        end
       
        loa = create_ap_loa(g1, grip_width);
        grasp_samples = {loa};
        [ mn_Q, v_Q, success, varargout] = mc_sample_fast(shape_params.all_points, ...
            cone_angle, grasp_samples, num_contacts, shape_samples, grid_dim, ...
            shape_params.surfaceThresh, bad_contact_thresh, plate_width, ...
            grip_width, false);
        
        grasps_scratch(j,:) = [loa(1,:), loa(2,:)];
        grasp_q_scratch(j) = mn_Q;
        
        if debug
            fprintf('Grasp %d Q: %f\n', j, mn_Q);
            figure(4);
            visualize_grasp(g1, shape_params, shape_image, scale, arrow_length, ...
                plate_width, grip_width);
            pause(1);
        end
    end
    grasp_q_sorted = sort(grasp_q_scratch, 'descend');
    k = 1;
    while k <= num_grasps_keep
       ind = find(grasp_q_scratch == grasp_q_sorted(k));
       l = 1;
       while l <= size(ind,2) && k <= num_grasps_keep
           grasps(i,k,:) = grasps_scratch(ind(l),:); 
           grasp_qualities(i,k) = grasp_q_scratch(ind(l));
           k = k+1;
           l = l+1;
       end
    end
end

%% save grasps

grasps_name = sprintf('%s/grasps_%d.mat', model_dir, downsample);
grasp_q_name = sprintf('%s/grasp_q_%d.mat', model_dir, downsample);

save(grasps_name, 'grasps');
save(grasp_q_name, 'grasp_qualities');
