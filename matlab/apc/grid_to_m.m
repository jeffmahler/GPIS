function y = grid_to_m(x, grid_res)
y = (x - ones(size(x))) * grid_res;
end

