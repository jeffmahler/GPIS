function [points , com] = create_shape(gridDim, image)
%CREATE_SHAPE Allows the user to create new shape through manual point 
% selection with a click GUI

% initialize empty objects
points = [];
button = 0;
scale = 1;

if ndims(image) > 2
    %[height, width, channels] = size(image);
    figure(1);
    imshow(image);
    [newX, newY, button] = ginput(2);
    x0 = min(newX);
    x1 = max(newX);
    y0 = min(newY);
    y1 = max(newY);
    w = x1 - x0;
    h = y1 - y0;
    dimension = max(w, h);
    scale = gridDim / dimension;
    image_resized = image(uint16(y0):uint16(y0+dimension), ...
        uint16(x0):uint16(x0+dimension), :);
end

figure(1);
cla;
if ndims(image) > 2
    imshow(image_resized);
else
    plot([], []);
    grid on;
    grid minor;
    xlim([1,gridDim]);
    ylim([1,gridDim]);  
end
set(gca,'YDir','reverse');

disp('Click points on shape surface in clockwise order. Press the x key when finished');

while button ~= 120
    [newX, newY, button] = ginput(1);
    if ndims(image) > 2
        newX = newX * scale;
        newY = newY * scale;
    end
    points = [points; newX, newY];

    % plot new shape
    if button ~= 120
        figure(1);
        if ndims(image) > 2
            imshow(image_resized);
            hold on;
            plot(points(:,1) / scale, points(:,2) / scale, 'r');
        else
            plot(points(:,1), points(:,2));
            grid on;
            grid minor;
            xlim([1,gridDim]);
            ylim([1,gridDim]);
        end
        set(gca,'YDir','reverse');
    end
end

% remove last point (its junk) and plot full shape
points = points(1:(size(points,1)-1),:);
points = [points; points(1,:)];
figure(1);
if ndims(image) > 2
    imshow(image_resized);
    hold on;
    plot(points(:,1) / scale, points(:,2) / scale, 'r');
else
    plot(points(:,1), points(:,2));
    grid on;
    grid minor;
    xlim([1,gridDim]);
    ylim([1,gridDim]);
end
set(gca,'YDir','reverse');

% fix shape
points = reshape(points', 1, 2*size(points,1));
points = round(points);

% get center of mass
disp('Click center of mass');
com = ginput(1);
com = com * scale;
com = round(com);
close;

end

