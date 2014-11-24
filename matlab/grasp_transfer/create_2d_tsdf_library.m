% Creates a 2D tsdf library given an initial directory

% define variables
data_dir = 'data/brown_dataset';
model_dir = 'data/grasp_transfer_models/brown_dataset';
downsample = 4;
padding = 20;
tsdf_thresh = 10;
vis_std = false;
bucket_size = 25;

%%
% enumerate directory
training_filenames = dir(data_dir);
training_filenames = training_filenames(3:end);
num_training = size(training_filenames,1);

% determine max size of images
max_dim = 0;
filenames = cell(num_training,1);

disp('Reading filenames for grid size');

for i = 1:num_training
    filename = sprintf('%s/%s', data_dir, training_filenames(i).name);
    filenames{i} = filename;
    shape_image = imread(filename);
   
    [height, width] = size(shape_image);
    big_dim = max(height, width);
    if big_dim > max_dim
        max_dim = big_dim;
    end
end

%% loop through the directory and load in each tsdf
grid_dim = max_dim + 2*padding;
ds_dim = uint16(grid_dim / downsample);

win = 7;
sigma_down = sqrt(2)^(downsample-1);
sigma_smooth = 1.0;
X = zeros(num_training, ds_dim^2);
h1 = fspecial('gaussian', win, sigma_down);
h2 = fspecial('gaussian', win, sigma_smooth);

for i = 1:num_training
    filename = filenames{i};
    if mod(i, 10) == 0
       fprintf('Reading shape file %d: %s\n', i, filename);
    end
    shape_image = imread(filename);
    
    M = 255*ones(grid_dim);
    M(padding+1:padding+size(shape_image,1), ...
      padding+1:padding+size(shape_image,2)) = shape_image;
    shape_image = M;
    
    tsdf = trunc_signed_distance(255-shape_image, tsdf_thresh);
    tsdf = imfilter(tsdf, h1);
    tsdf = imresize(tsdf, (double(ds_dim) / grid_dim));
    
    tsdf = standardize_tsdf(tsdf, vis_std);
    tsdf = imfilter(tsdf, h2);
    
    figure(1);
    imshow(tsdf);
    %pause(3);
    
    X(i,:) = tsdf(:)';
end

%% create KD-tree
kd_tree = KDTreeSearcher(X, 'BucketSize', bucket_size);

%% save everything
filenames_name = sprintf('%s/filenames.mat', model_dir);
tsdf_filename = sprintf('%s/tsdf_vectors_%d.mat', model_dir, downsample);
kd_tree_filename = sprintf('%s/kd_tree_%d.mat', model_dir, downsample);

save(filenames_name, 'filenames');
save(tsdf_filename, 'X');
save(kd_tree_filename, 'kd_tree');
