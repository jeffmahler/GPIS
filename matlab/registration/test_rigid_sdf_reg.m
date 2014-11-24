%% try out these Brown shapes
close all;

% shape1 = imread('data/pgms/hammer39.pgm');
% shape2 = imread('data/pgms/hammer40.pgm');
trans_scale = 2.0;
rot_scale = 10.0;
Z = eye(2) + rot_scale * rand(2);
[U, S, V] = svd(Z);
R = U * [1, 0; 0, det(U*V')] * V';
D = 0.25;
thresh = 10;

tf_true = struct();
tf_true.R = R;%eye(2);
tf_true.t = trans_scale * rand(2,1);
shape1 = imread('data/pgms/hammer16.pgm');
%gridDim = max([size(shape1) size(shape2)]);
gridDim = max(size(shape1));

M1 = 255*ones(gridDim);
M1(1:size(shape1,1), 1:size(shape1,2)) = shape1;
shape1 = M1;

% M2 = 255*ones(gridDim);
% M2(1:size(shape2,1), 1:size(shape2,2)) = shape2;
% shape2 = M2;

tsdf1 = trunc_signed_distance(255-shape1, thresh);
%tsdf2 = trunc_signed_distance(255-shape2, thresh);
tsdf2 = warp_grid(tf_true, tsdf1, 1.0);

figure;
subplot(1,2,1);
imagesc(tsdf1);
subplot(1,2,2);
imagesc(tsdf2);

%%
penalties = struct();
penalties.Q = 10.0;
stop_criteria = struct();
stop_criteria.T = 1.0;
stop_criteria.eps = 1e-2;
stop_criteria.max_iter = 20;
update_params = struct();
update_params.shrink_temp = 0.1;

tsdf1_down = imresize(tsdf1, D);
tsdf2_down = imresize(tsdf2, D);

%% register using our method
registration = register_2d_rigid_unknown_corrs(tsdf1_down, tsdf2_down, penalties,...
    stop_criteria, update_params);
registration.t = (1.0 / D) * registration.t;
 
%% interpolate both versions
tsdf1_reg = warp_grid(registration, tsdf1, 1.0);
interpolate_sdfs(tsdf1, tsdf2, gridDim, 0.5, 2);
interpolate_sdfs(tsdf1_reg, tsdf2, gridDim, 0.5, 2);

%% comparison with matlab
[opt, met] = imregconfig('monomodal');
tsdf1_padded = thresh * ones(3*gridDim); % pad image to remove warp fill-in
tsdf1_padded(gridDim+1:2*gridDim, gridDim+1:2*gridDim) = tsdf1;
tsdf1_reg = imregister(tsdf1_padded, tsdf2, 'similarity', opt, met);

figure(1);
subplot(1,3,1);
imagesc(tsdf1);
subplot(1,3,2);
imagesc(tsdf2);
subplot(1,3,3);
imagesc(tsdf1_reg);

interpolate_sdfs(tsdf1, tsdf2, gridDim, 0.5, 2);
interpolate_sdfs(tsdf1_reg, tsdf2, gridDim, 0.5, 2);

%% create registered tsdfs
%
%interpolate_sdfs(tsdf1_reg, tsdf2, gridDim, 0.5, 2)

% figure(1);
% subplot(1,2,1);
% imagesc(abs(tsdf1 - tsdf1_reg));
% subplot(1,2,2);
% imagesc(abs(tsdf1_reg - tsdf2));

figure(2);
subplot(1,3,1);
imagesc(tsdf1);
%sdf_surface(tsdf1, 0.5);
subplot(1,3,2);
imagesc(tsdf2);
%sdf_surface(tsdf2, 0.5);
subplot(1,3,3);
imagesc(tsdf1_reg);
%sdf_surface(tsdf1_reg, 0.5);
