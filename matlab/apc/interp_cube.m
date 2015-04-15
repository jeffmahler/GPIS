function interp_val = interp_cube( x, grid )
%INTERP_SQUARE Summary of this function goes here
%   Detailed explanation goes here

[height, width, depth] = size(grid);
x_min = max(floor(x), [1;1;1]);
x_max = min(ceil(x), [height; width; depth]);

interp_points = [x_min(1), x_min(2), x_min(3);
                 x_min(1), x_min(2), x_max(3);
                 x_min(1), x_max(2), x_min(3);
                 x_max(1), x_min(2), x_min(3);
                 x_min(1), x_max(2), x_max(3);
                 x_max(1), x_max(2), x_min(3);
                 x_max(1), x_min(2), x_max(3);
                 x_max(1), x_max(2), x_max(3)];
vals = zeros(8,1);
weights = ones(8,1);
for i = 1:8
    vals(i) = grid(interp_points(i,1), interp_points(i,2), interp_points(i,3));
    dist = norm(x - interp_points(i,:)');
    if dist > 0
        weights(i) = 1.0 / dist;
    end
end
if sum(weights) == 0
   weights = ones(8,1); 
end
weights = weights / sum(weights);
interp_val = weights' * vals;
    
end

