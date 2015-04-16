function points = get_points_gui(overlay_im)


if nargin < 1
    im_dim = [1000, 1000];
    overlay_im = ones(im_dim);
else
    im_dim = size(overlay_im);
end

figure(100);
imshow(overlay_im);
hold on;

disp('Click points. Press the X key when finished');

points = [];
term_sig = 0;
while term_sig ~= 120
    [point_x, point_y, term_sig] = ginput(1);
    if term_sig ~= 120
        points = [points; point_x, point_y];
        scatter(point_x, point_y, 'k');
    end
end

points(:,1) = points(:,1) / im_dim(2);
points(:,2) = points(:,2) / im_dim(1);

end

