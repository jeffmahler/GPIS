function grid_reg = warp_grid_3d(tf, grid, grid_center)

[height, width, depth] = size(grid);
centroid_scale = (1.0 / tf.s_center);
trans_scale = (1.0 / tf.s_trans);

grid_reg = zeros(height, width, depth);
for i = 1:height
    for j = 1:width
        for k = 1:depth
            y = [i; j; k];
            y = y - grid_center';
            x = tf.R' * (y - trans_scale * tf.t);
            x = grid_center' + centroid_scale * x;
            x(1) = max(min(x(1), double(height)), 1.0);
            x(2) = max(min(x(2), double(width)), 1.0);
            x(3) = max(min(x(3), double(depth)), 1.0);
            grid_reg(i,j,k) = interp_cube(x, grid);
        end
    end
end


end

