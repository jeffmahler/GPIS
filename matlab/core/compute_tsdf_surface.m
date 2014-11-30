function [tsdf_surface, varargout] = compute_tsdf_surface( tsdf, win )

if nargin < 2
   win = 3; 
end

tsdf_dims = size(tsdf);

% compute masks
inside_mask = tsdf < 0;
outside_mask = tsdf > 0;

% get surface points
SE = strel('square', win);
outside_di = imdilate(outside_mask, SE);
outside_mask_di = (outside_di == 1);
tsdf_surface = double(outside_mask_di & inside_mask);

surf_ind = find(tsdf_surface == 1);
inside_ind = find(inside_mask == 1);
outside_ind = find(outside_mask == 1);

if tsdf_dims(3) > 1
    [surf_x, surf_y, surf_z] = ind2sub(tsdf_dims, surf_ind);
    
    if nargout > 0
        varargout{1} = [surf_x(:) surf_y(:) surf_z(:)];
    end

    [inside_x, inside_y, inside_z] = ind2sub(tsdf_dims, inside_ind);
    if nargout > 1
        varargout{2} = [inside_x(:) inside_y(:) inside_z(:)];
    end

    [outside_x, outside_y, outside_z] = ind2sub(tsdf_dims, inside_ind);
    if nargout > 2
        varargout{3} = [outside_x(:) outside_y(:) outside_z(:)];
    end
else
    [surf_x, surf_y] = ind2sub(tsdf_dims, surf_ind); 
   
    if nargout > 0
        varargout{1} = [surf_x(:) surf_y(:)];
    end

    [inside_x, inside_y] = ind2sub(tsdf_dims, inside_ind);
    if nargout > 1
        varargout{2} = [inside_x(:) inside_y(:)];
    end

    [outside_x, outside_y] = ind2sub(tsdf_dims, outside_ind);
    if nargout > 2
        varargout{3} = [outside_x(:) outside_y(:)];
    end
end

end

