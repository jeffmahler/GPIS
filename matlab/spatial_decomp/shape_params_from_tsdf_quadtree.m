function shape_params = shape_params_from_tsdf_quadtree(tsdf, noise_params, min_dim, max_dim)

% compute tsdf gradient
[height, width] = size(tsdf);
grid_dim = height;
tsdf_grad = tsdf_gradient(tsdf);
Gx = tsdf_grad(:,:,1);
Gy = tsdf_grad(:,:,2);

% compute tsdf points
[X, Y] = meshgrid(1:grid_dim, 1:grid_dim);
points = [X(:), Y(:)];

% quad decomp the tsdf
[~, cell_centers, ~] = quadtree_decomp_tsdf(tsdf, min_dim, max_dim);

cell_centers_linear = cell_centers(2,:)' + ...
    (cell_centers(1,:)' - 1) * grid_dim;     

cell_normals = [Gx(cell_centers_linear), Gy(cell_centers_linear)];
cell_points = [X(cell_centers_linear), Y(cell_centers_linear)];

% add noise to tsdf
[measured_tsdf, noise_grid] = add_noise_to_tsdf(tsdf, cell_points, noise_params);
valid_indices = find(noise_grid < noise_params.occlusionScale);

shape_params = struct();
shape_params.gridDim = grid_dim;
shape_params.tsdf = measured_tsdf(valid_indices);
shape_params.normals = cell_normals(valid_indices,:);
shape_params.points = cell_points(valid_indices,:);
shape_params.noise = noise_grid(valid_indices);%noise_grid(:);
shape_params.all_points = [X(:) Y(:)];
shape_params.fullTsdf = tsdf(:);
shape_params.fullNormals = [Gx(:) Gy(:)];
shape_params.com = mean(shape_params.points(shape_params.tsdf < 0,:));


end

