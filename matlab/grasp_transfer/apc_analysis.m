% Code for running analyses on Amazon Picking Challenge items

%% read in sdf file
filename = 'data/completed_tsdf_texture_mapped_mesh.sdf';
sdf_file = textread(filename);
sdf_dims = sdf_file(1,:);
sdf_origin = sdf_file(2,:);
sdf_res = sdf_file(3,1);
sdf_vals = sdf_file(4:end,1);

sdf = reshape(sdf_vals, sdf_dims);

%% display raw sdf points
% sdf_thresh = 0.001;
% sdf_zc = find(abs(sdf) < sdf_thresh);
% [sdf_x, sdf_y, sdf_z] = ind2sub(sdf_dims, sdf_zc);
[sdf_surf_mask, surf_points, inside_points] = compute_tsdf_surface(sdf);
sdf_x = surf_points(:,1);
sdf_y = surf_points(:,2);
sdf_z = surf_points(:,3);
n_surf = size(surf_points,1);
centroid = mean(surf_points);

figure(1);
scatter3(sdf_x, sdf_y, sdf_z);
xlabel('X');
ylabel('Y');
zlabel('Z');
xlim([1, 100]);
ylim([1, 100]);
zlim([1, 100]);

%% display the octree and whatnot
pts = [sdf_x, sdf_y, sdf_z];
oct = OcTree(pts, 'binCapacity', 500);
figure
boxH = oct.plot; 
cols = lines(oct.BinCount); 
doplot3 = @(p,varargin) plot3(p(:,1), p(:,2), p(:,3), varargin{:}); 
for i = 1:oct.BinCount 
   set(boxH(i),'Color',cols(i,:),'LineWidth', 1+oct.BinDepths(i)) 
   doplot3(pts(oct.PointBins==i,:),'.','Color',cols(i,:)) 
end 
axis image, view(3)

%% get sdf gradients
sobel_3d = zeros(3,3,3);
sobel_2d = [-1 -2 -1;
             0  0  0;
             1  2  1];
sobel_3d(:,:,1) = sobel_2d;
sobel_3d(:,:,2) = 2*sobel_2d;
sobel_3d(:,:,3) = sobel_2d;

[Gx, Gy, Gz] = gradient(sdf);

%% get a random grasp
sigma_c = 5.0;
eps = 10.0;
arrow_length = 5;
step_size = 0.1;

% sample random spherical coords
theta = 2 * pi * rand();
phi = pi * rand();
r = max(max(surf_points - repmat(centroid, n_surf, 1))) + eps;

% sample grasp center
center = normrnd(centroid, sigma_c);

% convert spherical to cartesian
[g1_x, g1_y, g1_z] = sph2cart(theta, phi, r);
g1 = [g1_x g1_y g1_z] + center;
[g2_x, g2_y, g2_z] = sph2cart(theta, phi, -r);
g2 = [g2_x g2_y g2_z] + center;

grasp_diff = g2 - g1;
grasp_dir = grasp_diff / norm(grasp_diff);
start1 = g1 - arrow_length * grasp_dir;
start2 = g2 + arrow_length * grasp_dir;

g1_gp = [g1; g2];
g1_loa = compute_loa(g1_gp, step_size);

g2_gp = [g2; g1];
g2_loa = compute_loa(g2_gp, step_size);

loas = {g1_loa, g2_loa};
contacts = find_contacts(loas, sdf);

%% get grasp quality
friction_coef = 0.5;
n_cone_faces = 2;
n_contacts = 2;

[forces, failed] = compute_forces(contacts, Gx, Gy, Gz, friction_coef, n_cone_faces);
Q = ferrari_canny_3d(centroid', contacts', forces);

%% plot grasp stuff
figure(1);
clf;
scatter3(sdf_x, sdf_y, sdf_z);
hold on;
scatter3(g1_loa(:,1), g1_loa(:,2), g1_loa(:,3), 5, 'rx');
scatter3(contacts(:,1), contacts(:,2), contacts(:,3), 10, 'gx', 'LineWidth', 5);
arrow(start1, g1, 'FaceColor', 'r', 'EdgeColor', 'r', 'Length', 5, 'Width', 5, 'TipAngle', 45);
arrow(start2, g2, 'FaceColor', 'c', 'EdgeColor', 'c', 'Length', 5, 'Width', 5, 'TipAngle', 45);

for i = 1:n_contacts
    f = forces{i};
    n_forces = size(f, 2);
    contact = contacts(i,:);
   
    for j = 1:n_forces
        f_pt = contact - 10*f(:,j)';
        scatter3(f_pt(:,1), f_pt(:,2), f_pt(:,3), 10, 'mx', 'LineWidth', 5);
    end
end

% scatter3(g1(1), g1(2), g1(3), 100, 'rx', 'LineWidth', 5);
% scatter3(g2(1), g2(2), g2(3), 100, 'rx', 'LineWidth', 5);
xlabel('X');
ylabel('Y');
zlabel('Z');
xlim([1, 100]);
ylim([1, 100]);
zlim([1, 100]);


%% get the center points
bin_centers = octree_bin_centers(oct);

