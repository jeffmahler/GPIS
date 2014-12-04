function [qt, cell_centers, blocks] = quadtree_decomp_tsdf( tsdf, min_dim, max_dim )
%QUADTREE_DECOMP_TSDF Summary of this function goes here
%   Detailed explanation goes here

% setup
[height, width] = size(tsdf);
grid_dim = height;
outside_mask = tsdf > 0;

% compute quadtree decomposition
dim_diff = max_dim - grid_dim;
pad = floor(dim_diff / 2);
outside_mask_padded = ones(max_dim);
outside_mask_padded(pad+1:grid_dim+pad, ...
    pad+1:grid_dim+pad) = outside_mask;
qt = qtdecomp(outside_mask_padded, 0.1, [min_dim, max_dim]);

% get quadtree blocks
blocks = repmat(uint8(0), size(qt)); 

for dim = [4096 2048 1024 512 256 128 64 32 16 8 4 2 1];    
  numblocks = length(find(qt==dim));    
  if (numblocks > 0)        
    values = repmat(uint8(1),[dim dim numblocks]);
    values(2:dim,2:dim,:) = 0;
    blocks = qtsetblk(blocks,qt,dim,values);
  end
end

blocks(end,1:end) = 1;
blocks(1:end,end) = 1;

% parse cell centers
[pX, pY] = find(qt > 0);
num_cells = size(pX, 1);
cell_centers = zeros(2, num_cells);

for i = 1:num_cells
   p = [pY(i); pX(i)];
   cell_size = qt(p(1), p(2));
   cell_center = p + floor(cell_size / 2) * ones(2,1);
   cell_centers(:, i) = cell_center;
end

% snap cell centers to grid edges
cell_centers = cell_centers - pad;
cell_centers(cell_centers < 1) = 1;
cell_centers(cell_centers > grid_dim) = grid_dim;   

end

