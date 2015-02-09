function interp_val = interp_square( x, value_grid )
%INTERP_SQUARE Summary of this function goes here
%   Detailed explanation goes here

[height, width] = size(value_grid);
x_min = max(floor(x), [1;1]);
x_max = min(ceil(x), [width; height]);

interp_points = [x_min(1), x_min(2);
                 x_min(1), x_max(2);
                 x_max(1), x_min(2);
                 x_max(1), x_max(2)];
vals = zeros(4,1);
weights = ones(4,1);
for i = 1:4
    vals(i) = value_grid(interp_points(i,2), interp_points(i,1));
    dist = norm(x - interp_points(i,:)');
    if dist > 0
        weights(i) = 1.0 / dist;
    end
end
if sum(weights) == 0
   weights = ones(4,1); 
end
weights = weights / sum(weights);
interp_val = weights' * vals;
    
end

