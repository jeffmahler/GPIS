function [ ] = plot_grasp_2d(grasp, sdf, config )

% get params
sdf_dims = size(sdf);
arrow_length = config.arrow_length;
friction_coef = config.friction_coef;
n_cone_faces = config.n_cone_faces;
scale = config.scale;
grasp_width = config.grasp_width;
plate_width = config.plate_width;

if ~grasp.constrained_2d
    return;
end

% get slice of grasp
grasp_slice_axis = grasp.slice;
grasp_slice = grasp.g1(grasp_slice_axis);
if grasp_slice_axis == 1
    sdf_slice = sdf(grasp_slice, :, :);
    g1_open = [grasp.g1_open(2), grasp.g1_open(3)];
    g2_open = [grasp.g2_open(2), grasp.g2_open(3)];
    g1 = [grasp.g1(2), grasp.g1(3)];
    g2 = [grasp.g2(2), grasp.g2(3)];
    
elseif grasp_slice_axis == 2
    sdf_slice = sdf(:, grasp_slice, :);
    g1_open = [grasp.g1_open(1), grasp.g1_open(3)];
    g2_open = [grasp.g2_open(1), grasp.g2_open(3)];
    g1 = [grasp.g1(1), grasp.g1(3)];
    g2 = [grasp.g2(1), grasp.g2(3)];
else
    sdf_slice = sdf(:, :, grasp_slice);
    g1_open = [grasp.g1_open(1), grasp.g1_open(2)];
    g2_open = [grasp.g2_open(1), grasp.g2_open(2)];
    g1 = [grasp.g1(1), grasp.g1(2)];
    g2 = [grasp.g2(1), grasp.g2(2)];
end

% get new sdf params
sdf_slice = squeeze(sdf_slice);
sdf_dims = size(sdf_slice);
sdf_slice_resized = sdf_slice;
se = strel('square', 5);

% high res image
s = 2;
while s < scale
    sdf_slice_resized = imresize(sdf_slice_resized, 2, 'bilinear');
    s = min(2 * s, scale);
end
final_resize = scale * sdf_dims(1) / size(sdf_slice_resized, 1);
sdf_slice_resized = imresize(sdf_slice_resized, final_resize, 'bilinear');
exterior_image = 255 * double(sdf_slice_resized > 0);
exterior_image = imopen(exterior_image, se);

imshow(exterior_image);
hold on;

% setup grasp vars
grasp_dir = g2 - g1;
grasp_dir = grasp_dir / norm(grasp_dir);

% (HACK) need to run twice bc neither are in contact...
[contacts, ~] = ...
    antipodal_grasp_contacts(scale*g1_open, grasp_dir, sdf_slice_resized, scale*grasp_width);
[contacts, contact_found] = ...
    antipodal_grasp_contacts(contacts(2,:), -grasp_dir, sdf_slice_resized, scale*grasp_width);
if ~contact_found 
    return;
end

% get start dirs
g1_im = contacts(2,:);
g2_im = contacts(1,:);
start1 = g1_im - scale * arrow_length * grasp_dir;
start2 = g2_im + scale * arrow_length * grasp_dir;

% scatter points, contacts, etc
arrow([start1(2) start1(1)], [g1_im(2), g1_im(1)], 'FaceColor', 'r', 'EdgeColor', 'r', 'Length', 12, 'Width', 8, 'TipAngle', 30);
arrow([start2(2) start2(1)], [g2_im(2), g2_im(1)], 'FaceColor', 'r', 'EdgeColor', 'r', 'Length', 12, 'Width', 8, 'TipAngle', 30);

% plot fake plates
tan1 = [grasp_dir(2) -grasp_dir(1)];
tan2 = [grasp_dir(2) -grasp_dir(1)];
plate_width = scale * double(plate_width);

start1 = g1_im + plate_width * tan1 / 2;
start2 = g2_im + plate_width * tan2 / 2;
end1 = g1_im - plate_width * tan1 / 2;
end2 = g2_im - plate_width * tan2 / 2;

plot([start1(2); end1(2)], [start1(1); end1(1)], 'r', 'LineWidth', 8);
plot([start2(2); end2(2)], [start2(1); end2(1)], 'r', 'LineWidth', 8);

end