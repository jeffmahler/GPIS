% test quadtree decomposition
close all

%test_dims = [16, 32, 64];
test_dims = [64, 128, 256, 512, 1024, 2048];
num_tests = size(test_dims, 2);
percentages = zeros(1, num_tests);
quad_times = zeros(1, num_tests);
full_construction_times = zeros(1, num_tests);
quad_construction_times = zeros(1, num_tests);
full_mse = zeros(1, num_tests);
quad_mse = zeros(1, num_tests);
full_surf_mse = zeros(1, num_tests);
quad_surf_mse = zeros(1, num_tests);
min_dim = 2;
max_dim = 128;
grid_dim = size(tsdf1, 1);

test_gpis_contstruction = false;

% config for GPIS construction
dummy_full_filename = 'full';
dummy_quad_filename = 'quad';
data_dir = 'results/quadtree';
num_samples = 1;
scale = 1.0;
noise_scale = 0.1;

training_params = struct();
training_params.activeSetMethod = 'Full';
training_params.activeSetSize = 1;
training_params.beta = 10;
training_params.firstIndex = 150;
training_params.numIters = 0;
training_params.eps = 1e-2;
training_params.delta = 1e-2;
training_params.levelSet = 0;
training_params.surfaceThresh = 0.1;
training_params.scale = scale;
training_params.numSamples = 20;
training_params.trainHyp = false;
training_params.hyp = struct();
training_params.hyp.cov = [log(exp(1)), log(1)];
training_params.hyp.mean = [0; 0; 0];
training_params.hyp.lik = log(0.1);
training_params.useGradients = true;

shape_params = struct();

for k = 1:num_tests

    max_dim = test_dims(k);
    tsdf_resized = imresize(tsdf1, max_dim / grid_dim);
    inside_mask = tsdf_resized < 0;
    outside_mask = tsdf_resized > 0;
    
    % get surface points
    SE = strel('square', 3);
    outside_di = imdilate(outside_mask, SE);
    outside_mask_di = (outside_di== 1);
    tsdf_surface = double(outside_mask_di & inside_mask);
    tsdf_surf_points = find(tsdf_surface(:) == 1);
    
    start_time = tic;
    S = qtdecomp(outside_mask, 0.1, [min_dim, max_dim]);
    blocks = repmat(uint8(0),size(S));
    quad_times(k) = toc(start_time); 

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

    figure(10); 
    subplot(1,2,1);
    imshow(tsdf_resized);
    subplot(1,2,2);
    imshow(blocks,[]);

    % get centers of tsdf cells
    [X, Y] = find(S > 0);
    num_cells = size(X, 1);
    cell_centers = zeros(2, num_cells);

    for i = 1:num_cells
       p = [Y(i); X(i)];
       cell_size = S(p(1), p(2));
       cell_center = p + floor(cell_size / 2) * ones(2,1);
       cell_centers(:, i) = cell_center;
    end
    %cell_centers = uint16(cell_centers);

    % plot data points
    figure(11);
    scatter(cell_centers(1,:), cell_centers(2,:));
    set(gca,'YDir','Reverse')

    percentages(k) = (num_cells / max_dim);
    % fprintf('TSDF of dim %d reduced to %f%% of original data size\n', ...
    %    max_dim, percentages(k));
    
    if test_gpis_contstruction
        % construct GPIS using full and quadtree points
        [Gx, Gy] = imgradientxy(tsdf_resized, 'CentralDifference');
        [X, Y] = meshgrid(1:max_dim, 1:max_dim);
        noise_grid = noise_scale * ones(max_dim);

        shape_params.gridDim = max_dim;
        shape_params.tsdf = tsdf_resized(:);
        shape_params.normals = [Gx(:), Gy(:)];
        shape_params.points = [X(:), Y(:)];
        shape_params.noise = noise_grid(:);
        shape_params.all_points = shape_params.points;
        shape_params.fullTsdf = shape_params.tsdf;
        shape_params.fullNormals = shape_params.normals;
        shape_params.com = mean(shape_params.points(shape_params.tsdf < 0,:));

        [fullGpModel, fullShapeSamples, fullConstructionResults] = ...
            construct_and_save_gpis(dummy_full_filename, data_dir, shape_params, ...
                                    training_params, num_samples, scale);
        full_construction_times(k) = fullConstructionResults.constructionTime;
        full_mse(k) = mean((fullConstructionResults.predGrid.tsdf(:) - tsdf_resized(:)).^2);
        full_surf_mse(k) = mean((fullConstructionResults.predGrid.tsdf(tsdf_surf_points) ...
            - tsdf_resized(tsdf_surf_points)).^2);

        % quadtree gpis
        cell_centers_linear = cell_centers(2,:)' + cell_centers(1,:)' * max_dim;
        shape_params.tsdf = tsdf_resized(cell_centers_linear);
        shape_params.normals = [Gx(cell_centers_linear), Gy(cell_centers_linear)];
        shape_params.points = cell_centers';
        shape_params.noise = noise_grid(cell_centers_linear);

        [quadGpModel, quadShapeSamples, quadConstructionResults] = ...
            construct_and_save_gpis(dummy_quad_filename, data_dir, shape_params, ...
                                    training_params, num_samples, scale);
        quad_construction_times(k) = quadConstructionResults.constructionTime;
        quad_mse(k) = mean((quadConstructionResults.predGrid.tsdf(:) - tsdf_resized(:)).^2);
        quad_surf_mse(k) = mean((quadConstructionResults.predGrid.tsdf(tsdf_surf_points) ...
            - tsdf_resized(tsdf_surf_points)).^2);

        full_pred_grid = reshape(fullConstructionResults.predGrid.tsdf, [max_dim, max_dim]);
        quad_pred_grid = reshape(quadConstructionResults.predGrid.tsdf, [max_dim, max_dim]);
        figure(6);
        subplot(1,4,1);
        imagesc(quad_pred_grid);
        subplot(1,4,2);
        imagesc(tsdf_resized);
        subplot(1,4,3);
        imagesc(full_pred_grid);
        subplot(1,4,4);
        imagesc(tsdf_surface);
    end
end

figure(7);
plot(test_dims, percentages, 'b-o', 'LineWidth', 2);
title('Percentage of TSDF points used in quadtree vs TSDF dimension', 'FontSize', 15);
xlabel('TSDF dimension (px)', 'FontSize', 15);
ylabel('# Quadtree Cells', 'FontSize', 15);

figure(8);
plot(test_dims, quad_times, 'b-o', 'LineWidth', 2);
title('Quadtree construction time vs TSDF dimension', 'FontSize', 15);
xlabel('TSDF dimension (px)', 'FontSize', 15);
ylabel('Quadtree Time (sec)', 'FontSize', 15);

if test_gpis_contstruction
    figure(9);
    plot(test_dims, full_construction_times, 'b-o', 'LineWidth', 2);
    hold on;
    plot(test_dims, quad_construction_times + quad_times, 'g-o', 'LineWidth', 2);
    title('GPIS construction time vs TSDF dimension', 'FontSize', 15);
    xlabel('TSDF dimension (px)', 'FontSize', 15);
    ylabel('Quadtree Time (sec)', 'FontSize', 15);
    legend('Full', 'Quadtree', 'Location', 'Best');

    figure(16);
    plot(test_dims, full_surf_mse, 'b-o', 'LineWidth', 2);
    hold on;
    plot(test_dims, quad_surf_mse, 'g-o', 'LineWidth', 2);
    title('MSE of TSDF Surface vs TSDF dimension', 'FontSize', 15);
    xlabel('TSDF dimension (px)', 'FontSize', 15);
    ylabel('MSE (dist^2)', 'FontSize', 15);
    legend('Full', 'Quadtree', 'Location', 'Best');
end