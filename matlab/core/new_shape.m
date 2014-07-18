function [points, com] = new_shape(filename, dataDir, dim)
% Create and save a new shape
[points, com] = create_shape(dim);

pointsName = sprintf('%s/%s_points.csv', dataDir, filename);
comName = sprintf('%s/%s_com.csv', dataDir, filename);
csvwrite(pointsName, points);
csvwrite(comName, com);

end

