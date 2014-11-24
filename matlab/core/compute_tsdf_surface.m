function [tsdf_surface, varargout] = compute_tsdf_surface( tsdf, win )

if nargin < 2
   win = 3; 
end

% compute masks
inside_mask = tsdf < 0;
outside_mask = tsdf > 0;

% get surface points
SE = strel('square', win);
outside_di = imdilate(outside_mask, SE);
outside_mask_di = (outside_di == 1);
tsdf_surface = double(outside_mask_di & inside_mask);

[surf_x, surf_y] = find(tsdf_surface == 1);
if nargout > 0
    varargout{1} = [surf_x(:) surf_y(:)];
end

[inside_x, inside_y] = find(inside_mask == 1);
if nargout > 1
    varargout{2} = [inside_x(:) inside_y(:)];
end

[outside_x, outside_y] = find(outside_mask == 1);
if nargout > 2
    varargout{3} = [outside_x(:) outside_y(:)];
end


end

