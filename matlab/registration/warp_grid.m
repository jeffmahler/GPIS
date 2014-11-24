function grid_reg = warp_grid(tf, grid, grid_center)

[height, width] = size(grid);
centroid_scale = (1.0 / tf.s_center);
trans_scale = (1.0 / tf.s_trans);

grid_reg = zeros(height, width);
for i = 1:height
    for j = 1:width
        y = [j;i];
        x = tf.R' * (y - trans_scale * tf.t);
        x = grid_center' + centroid_scale * (x - grid_center');
        x(1) = max(min(x(1), double(width)), 1.0);
        x(2) = max(min(x(2), double(height)), 1.0);
        grid_reg(i,j) = interp_square(x, grid);
    end
end


end

