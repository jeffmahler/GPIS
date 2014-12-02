function tsdf = tsdf_from_occupancy_grid(occupancy_grid, padding, ...
   tsdf_thresh, tsdf_smoothing_filter, out_size, vis_std)

[height, width] = size(occupancy_grid);
grid_dim = height;

% add padding
M = zeros(grid_dim+2*padding);
M(padding+1:padding+height, ...
  padding+1:padding+width) = occupancy_grid;
occupancy_grid = M;
occupancy_grid = imresize(occupancy_grid, (out_size / size(M,1)));
occupancy_grid = occupancy_grid > 0.5;

% compute sdf
tsdf = trunc_signed_distance(occupancy_grid, tsdf_thresh);

% create main parameters
tsdf = standardize_tsdf(tsdf, vis_std);
tsdf = imfilter(tsdf, tsdf_smoothing_filter);

end

