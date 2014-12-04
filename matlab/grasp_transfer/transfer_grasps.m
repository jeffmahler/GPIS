% Generate a new uncertain shape and lookup similar items in the database
data_dir = 'data/grasp_transfer_models/test';
shape_index = 8576;
shape_names = {'mandolin'};
grip_scales = {0.4};
tsdf_thresh = 10;
padding = 5;
scale = 4;

shape_indices = [8576];

% get random shape indices
num_test_shapes = 1000;
shape_indices = round(8600 * rand(num_test_shapes, 1) + 1);

config = struct();
config.arrow_length = 10;
config.scale = 1.0;
config.friction_coef = 0.5;
config.plate_width = 3;
config.grip_scale = 0.4;
config.padding = 5;
config.tsdf_thresh = 10;
config.downsample = 4;

config.test_data_file = 'data/caltech/caltech101_silhouettes_28.mat';
config.mat_save_dir = 'data/grasp_transfer_models/test';
config.model_dir = 'data/grasp_transfer_models/brown_dataset';

config.filter_win = 7;
config.filter_sigma = sqrt(2)^2;
config.vis_std = false;

config.quad_min_dim = 2;
config.quad_max_dim = 128;

config.num_shape_samples = 100;
config.image_scale = 4.0;

config.knn = 16;
config.vis_knn = false;
config.num_grasp_samples = 1500;

caltech_data = load(config.test_data_file);
num_points = size(caltech_data.X, 2);
data_dim = sqrt(num_points);

fprintf('Shape type: %s\n', caltech_data.classnames{caltech_data.Y(shape_index)});

%% load database into memtory
model_dir = 'data/grasp_transfer_models/brown_dataset';
downsample = 4;

% load models
filenames_filename = sprintf('%s/filenames.mat', model_dir);
tsdf_filename = sprintf('%s/tsdf_vectors_%d.mat', model_dir, downsample);
kd_tree_filename = sprintf('%s/kd_tree_%d.mat', model_dir, downsample);
grasps_filename = sprintf('%s/grasps_%d.mat', model_dir, downsample);
grasp_q_filename = sprintf('%s/grasp_q_%d.mat', model_dir, downsample);

S = load(filenames_filename);
filenames = S.filenames;

S = load(tsdf_filename);
tsdf_vectors = S.X;

S = load(kd_tree_filename);
kd_tree = S.kd_tree;

S = load(grasps_filename);
grasps = S.grasps;

S = load(grasp_q_filename);
grasp_q = S.grasp_qualities;

num_training = size(tsdf_vectors, 1);
num_points = size(tsdf_vectors, 2);
grid_dim = sqrt(num_points);

%% construct gpis
noise_scale = 0.1;
win = 7;
sigma = sqrt(2)^2;
h = fspecial('gaussian', win, sigma);
vis_std = true;

outside_mask = caltech_data.X(shape_index,:);
outside_mask = reshape(outside_mask, [data_dim, data_dim]);
M = ones(data_dim+2*padding);
M(padding+1:padding+size(outside_mask,1), ...
  padding+1:padding+size(outside_mask,2)) = outside_mask;
outside_mask = M;
outside_mask = imresize(outside_mask, (double(grid_dim) / size(M,1)));
outside_mask = outside_mask > 0.5;
tsdf = trunc_signed_distance(1-outside_mask, tsdf_thresh);

% create main parameters
tsdf = standardize_tsdf(tsdf, vis_std);
tsdf = imfilter(tsdf, h);

[Gx, Gy] = imgradientxy(tsdf, 'CentralDifference');
tsdf_grad = zeros(grid_dim, grid_dim, 2);
tsdf_grad(:,:,1) = Gx;
tsdf_grad(:,:,2) = Gy;

[X, Y] = meshgrid(1:grid_dim, 1:grid_dim);
points = [X(:), Y(:)];

% quadtree
min_dim = 2;
max_dim = 128;
inside_mask = tsdf < 0;
outside_mask = tsdf > 0;
[tsdf_surface, tsdf_surf_points, inside_points, outside_points] = ...
    compute_tsdf_surface(tsdf);

dim_diff = max_dim - grid_dim;
pad = floor(dim_diff / 2);
outside_mask_padded = ones(max_dim);
outside_mask_padded(pad+1:grid_dim+pad, ...
    pad+1:grid_dim+pad) = outside_mask;
S = qtdecomp(outside_mask_padded, 0.1, [min_dim, max_dim]);
blocks = repmat(uint8(0),size(S)); 

for dim = [2048 1024 512 256 128 64 32 16 8 4 2 1];    
  numblocks = length(find(S==dim));    
  if (numblocks > 0)        
    values = repmat(uint8(1),[dim dim numblocks]);
    values(2:dim,2:dim,:) = 0;
    blocks = qtsetblk(blocks,S,dim,values);
  end
end

blocks(end,1:end) = 1;
blocks(1:end,end) = 1;

% parse cell centers
[pX, pY] = find(S > 0);
num_cells = size(pX, 1);
cell_centers = zeros(2, num_cells);

for i = 1:num_cells
   p = [pY(i); pX(i)];
   cell_size = S(p(1), p(2));
   cell_center = p + floor(cell_size / 2) * ones(2,1);
   cell_centers(:, i) = cell_center;
end

% trim cell centers, add noise
%% variance parameters
var_params = struct();
var_params.y_thresh1_low = 79;
var_params.y_thresh1_high = 79;
var_params.x_thresh1_low = 79;
var_params.x_thresh1_high = 79;

var_params.y_thresh2_low = 79;
var_params.y_thresh2_high = 79;
var_params.x_thresh2_low = 79;
var_params.x_thresh2_high = 79;

var_params.y_thresh3_low = 79;
var_params.y_thresh3_high = 79;
var_params.x_thresh3_low = 79;
var_params.x_thresh3_high = 79;

var_params.occ_y_thresh1_low = 40;
var_params.occ_y_thresh1_high = 79;
var_params.occ_x_thresh1_low = 1;
var_params.occ_x_thresh1_high = 79;

var_params.occ_y_thresh2_low = 79;
var_params.occ_y_thresh2_high = 79;
var_params.occ_x_thresh2_low = 79;
var_params.occ_x_thresh2_high = 79;

var_params.transp_y_thresh1_low = 79;
var_params.transp_y_thresh1_high = 79;
var_params.transp_x_thresh1_low = 79;
var_params.transp_x_thresh1_high = 79;

var_params.transp_y_thresh2_low = 79;
var_params.transp_y_thresh2_high = 79;
var_params.transp_x_thresh2_low = 79;
var_params.transp_x_thresh2_high = 79;

var_params.occlusionScale = 1000;
var_params.transpScale = 4.0;
var_params.noiseScale = 0.2;
var_params.interiorRate = 0.1;
var_params.specularNoise = true;
var_params.sparsityRate = 0.2;
var_params.sparseScaling = 1000;
var_params.edgeWin = 2;

var_params.noiseGradMode = 'None';
var_params.horizScale = 1;
var_params.vertScale = 1;

config.noise_params = var_params;

cell_centers_mod = cell_centers - pad;
cell_centers_mod(cell_centers_mod < 1) = 1;
cell_centers_mod(cell_centers_mod > grid_dim) = grid_dim; 
cell_centers_linear = cell_centers_mod(2,:)' + ...
    (cell_centers_mod(1,:)' - 1) * grid_dim;     
num_centers = size(cell_centers_mod, 2);
noise = zeros(num_centers, 1);
measured_tsdf = tsdf(cell_centers_linear);

for k = 1:num_centers
    i = cell_centers_mod(2,k);
    j = cell_centers_mod(1,k);
    i_low = max(1,i-var_params.edgeWin);
    i_high = min(grid_dim,i+var_params.edgeWin);
    j_low = max(1,j-var_params.edgeWin);
    j_high = min(grid_dim,j+var_params.edgeWin);
    tsdf_win = tsdf(i_low:i_high, j_low:j_high);
  
    % add in transparency, occlusions
    if ((i > var_params.transp_y_thresh1_low && i <= var_params.transp_y_thresh1_high && ...
          j > var_params.transp_x_thresh1_low && j <= var_params.transp_x_thresh1_high) || ...
          (i > var_params.transp_y_thresh2_low && i <= var_params.transp_y_thresh2_high && ...
          j > var_params.transp_x_thresh2_low && j <= var_params.transp_x_thresh2_high) )
        % occluded regions
        if tsdf(i,j) < 0.6 % only add noise to ones that were actually in the shape
            measured_tsdf(k) = 0.5; % set outside shape
            noise(k) = var_params.transpScale; 
        end

    elseif min(min(tsdf_win)) < 0.6 && ((i > var_params.y_thresh1_low && i <= var_params.y_thresh1_high && ...
            j > var_params.x_thresh1_low && j <= var_params.x_thresh1_high) || ...
            (i > var_params.y_thresh2_low && i <= var_params.y_thresh2_high && ... 
            j > var_params.x_thresh2_low && j <= var_params.x_thresh2_high) || ...
            (i > var_params.y_thresh3_low && i <= var_params.y_thresh3_high && ... 
            j > var_params.x_thresh3_low && j <= var_params.x_thresh3_high))

        noise(k) = var_params.occlusionScale;
    elseif ((i > var_params.occ_y_thresh1_low && i <= var_params.occ_y_thresh1_high && ...
            j > var_params.occ_x_thresh1_low && j <= var_params.occ_x_thresh1_high) || ... 
            (i > var_params.occ_y_thresh2_low && i <= var_params.occ_y_thresh2_high && ...
            j > var_params.occ_x_thresh2_low && j <= var_params.occ_x_thresh2_high) )
        % occluded regions
        noise(k) = var_params.occlusionScale;

    elseif tsdf(i,j) < -0.5 % only use a few interior points (since realistically we wouldn't measure them)
        if rand() > (1-var_params.interiorRate)
           noise(k) = var_params.noiseScale;
        else
           noise(k) = var_params.occlusionScale; 
        end
    else
        noise_val = 1; % scaling for noise

        % add specularity to surface
        if var_params.specularNoise && min(min(abs(tsdf_win))) < 0.6
            noise_val = rand();

            if rand() > (1-var_params.sparsityRate)
                noise_val = var_params.occlusionScale / var_params.noiseScale; % missing data not super noisy data
                %noiseVal = noiseVal * varParams.sparseScaling;
            end
        end
        noise(k) = noise_val * var_params.noiseScale;
    end
end
%noise_grid = noise_scale * ones(grid_dim);

cell_normals = [Gx(cell_centers_linear), Gy(cell_centers_linear)];
cell_points = [X(cell_centers_linear), Y(cell_centers_linear)];
valid_indices = find(noise < var_params.occlusionScale);

shape_params = struct();
shape_params.gridDim = grid_dim;
shape_params.tsdf = measured_tsdf(valid_indices);
shape_params.normals = cell_normals(valid_indices,:);
shape_params.points = cell_points(valid_indices,:);
shape_params.noise = noise(valid_indices);%noise_grid(:);
shape_params.all_points = [X(:) Y(:)];
shape_params.fullTsdf = tsdf(:);
shape_params.fullNormals = [Gx(:) Gy(:)];
shape_params.com = mean(shape_params.points(shape_params.tsdf < 0,:));
shape_params.surfaceThresh = 0.1;
% figure(11);
% scatter(shape_params.points(:,1), shape_params.points(:,2));
% set(gca,'YDir','Reverse');

construction_params = struct();
construction_params.activeSetMethod = 'Full';
construction_params.activeSetSize = 1;
construction_params.beta = 10;
construction_params.firstIndex = 150;
construction_params.numIters = 0;
construction_params.eps = 1e-2;
construction_params.delta = 1e-2;
construction_params.levelSet = 0;
construction_params.surfaceThresh = 0.1;
construction_params.scale = 1.0;
construction_params.numSamples = 20;
construction_params.trainHyp = false;
construction_params.hyp = struct();
construction_params.hyp.cov = [log(exp(2)), log(1)];
construction_params.hyp.mean = [0; 0; 0];
construction_params.hyp.lik = log(0.1);
construction_params.useGradients = true;
construction_params.downsample = 2;

config.construction_params = construction_params;

num_samples = 100;
image_scale = 4.0;
[gp_model, shape_samples, construction_results] = ...
            construct_and_save_gpis(shape_names{1}, data_dir, shape_params, ...
                                    construction_params, num_samples, image_scale);
        

%% lookup nearest neighbors
K = 16;
idx = knnsearch(kd_tree, tsdf(:)', 'K', K);

grasp = zeros(4,1);
grasp_candidates = zeros(K, 4);
    
if config.vis_knn
    figure(66);
    imshow(tsdf);
    title('Original TSDF');

    figure(77);
    tsdf_grad_neigh = zeros(grid_dim, grid_dim, 2);
end

% tsdf_corr_thresh = 2.0;
% alpha = 1.0;
% beta = 1.0;
% corr_win = 3;

for i = 1:K
    % load neighbor attributes
    tsdf_neighbor = tsdf_vectors(idx(i),:);
    grasps_neighbor = grasps(idx(i), :, :);
    grasp_q_neighbor = grasp_q(idx(i), :);
    tsdf_neighbor = reshape(tsdf_neighbor, [grid_dim, grid_dim]);
    [Gx_nbr, Gy_nbr] = imgradientxy(tsdf_neighbor, 'CentralDifference');
    tsdf_grad_nbr(:,:,1) = Gx_nbr;
    tsdf_grad_nbr(:,:,2) = Gy_nbr;
   
    % tps registration
%    [corrs, outliers] = tsdf_tps_corrs(points, tsdf_neighbor, tsdf, tsdf_grad_nbr, tsdf_grad,...
%        tsdf_corr_thresh, alpha, beta, corr_win);
%     source_points = points(corrs(~outliers),:);
%     target_points = points(~outliers,:);
%     st = tpaps(target_points', source_points');
   
    % transfer neighbor grasp
    neighbor_shape_params = tsdf_to_shape_params(tsdf_neighbor);
    neighbor_shape_params.surfaceThresh = 0.1;

    for j = 1:1
        grasp(:) = grasps_neighbor(1,j,:);
        grasp_candidates(i, :) = grasp';
        
%         subplot(sqrt(25),sqrt(25),j);
%         visualize_grasp(grasp, neighbor_shape_params, tsdf_neighbor, ...
%             config.scale, config.arrow_length, config.plate_width, grid_dim);
%         title(sprintf('Grasp %d', j));
    end
    % visualization
    if i == 11
        test = 1;
    end
    subplot(sqrt(K),sqrt(K),i);
    visualize_grasp(grasp, neighbor_shape_params, tsdf_neighbor, ...
         config.scale, config.arrow_length, config.plate_width, grid_dim);
    title(sprintf('Neighbor %d', i));
end

%% use bandits to select the best grasp

grasp_samples = collect_samples_grasps(gp_model, grasp_candidates,1500,config,shape_params);
best_grasp = monte_carlo(grasp_samples, K, shape_params, config, tsdf);


%%
close all; 
best_grasp = succesive_rejects(grasp_samples, K, shape_params, config, tsdf);
