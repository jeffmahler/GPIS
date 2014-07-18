function [points , com] = create_shape(gridDim)
%CREATE_SHAPE Allows the user to create new shape through manual point 
% selection with a click GUI

% initialize empty objects
points = [];
button = 0;

figure(1);
cla;
plot([], []);
grid on;
grid minor;
xlim([1,gridDim]);
ylim([1,gridDim]);
set(gca,'YDir','reverse');

disp('Click points on shape surface in clockwise order. Press the x key when finished');

while button ~= 120
    [newX, newY, button] = ginput(1);
    points = [points; newX, newY];

    % plot new shape
    if button ~= 120
        figure(1);
        plot(points(:,1), points(:,2));
        grid on;
        grid minor;
        xlim([1,gridDim]);
        ylim([1,gridDim]);
        set(gca,'YDir','reverse');
    end
end

% remove last point (its junk) and plot full shape
points = points(1:(size(points,1)-1),:);
points = [points; points(1,:)];
figure(1);
plot(points(:,1), points(:,2));
grid on;
grid minor;
xlim([1,gridDim]);
ylim([1,gridDim]);
set(gca,'YDir','reverse');

% fix shape
points = reshape(points', 1, 2*size(points,1));
points = round(points);

% get center of mass
disp('Click center of mass');
com = ginput(1);
com = round(com);
close;

end

