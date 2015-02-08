function grid_reg = warp_mean_function(tf, grid, grid_center)

[height, width] = size(grid);

grid_reg = zeros(height, width);
for i = 1:height
    for j = 1:width
        y = [j;i];
        y = y - grid_center';
        x = (1.0 / tf.s) * (tf.R' * (y - tf.t));
        x = x + (1.0 / tf.s) * grid_center';
        x(1) = max(min(x(1), double(width)), 1.0);
        x(2) = max(min(x(2), double(height)), 1.0);
        grid_reg(i,j) = interp_square(x, grid);
    end
end


end

