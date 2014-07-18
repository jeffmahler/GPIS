function points = create_shape(gridDim)
%CREATE_SHAPE Allows the user to create new shape through manual point 
% selection with a click GUI

% initialize empty objects
points = [];
button = 0;

figure(1);
cla;
plot([], []);
xlim([1,gridDim]);
ylim([1,gridDim]);
set(gca,'YDir','reverse');

disp('Click points on shape surface in clockwise order. Press the x key when finished');

while button ~= 120
    [newX, newY, button] = ginput(1);
    points = [points; newX, newY]

    % plot new shape
    figure(1);
    plot(points(:,1), points(:,2));
    xlim([1,gridDim]);
    ylim([1,gridDim]);
    set(gca,'YDir','reverse');
end

points = points(1:(size(points,1)-1),:);
points = reshape(points', 1, 2*size(points,1));
points = round(points);
close;

end

