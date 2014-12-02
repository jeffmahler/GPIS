function [cp, cp_mc] = get_random_grasp(grid_dim, grid_dim_shape, random, sigma_c)
%Returns a random grasp specified by a line of action with start and end
%points 

if nargin < 3
    random = 0;
end
if nargin < 4
    sigma_c = 5.0;
end


%TODO: Support more than 2 contact points
grid_center = grid_dim / 2;
theta = rand() * 2 * pi;
r = grid_dim / 2;

center = normrnd(grid_center, sigma_c);
[g1_x, g1_y, g1_z] = sph2cart(theta, 0, r);
g1 = [g1_x g1_y] + center;
[g2_x, g2_y, g2_z] = sph2cart(theta, 0, -r);
g2 = [g2_x g2_y] + center;

cp = [g1; g2; g2; g1];
cp_mc = [g1; g2; g2; g1];

% theta1 = rand()*pi; 
% %theta1 = 0; 
% theta2 = rand()*pi+pi;
% 
% trans = grid_dim/2;
% trans_shape = grid_dim_shape/2;
% 
% grid_dim = grid_dim/2;
% grid_dim_shape = grid_dim_shape/2;
% 
%  if(random) 
%      x_trans = rand()*1/2;
%      y_trans = rand()*1/2; 
%  else
%      x_trans = 0; 
%      y_trans = 0;
%  end
% 
%  cp(1,:) = [cos(theta1)+x_trans sin(theta1)+y_trans]; 
%  cp(2,:) = [-cos(theta1)+x_trans -sin(theta1)+y_trans]; 
%     
%  cp(1,:) = grid_dim*cp(1,:)/norm(cp(1,:))+trans;
%  cp(2,:) = grid_dim*cp(2,:)/norm(cp(2,:))+trans;
% 
%  cp(3,:) = cp(2,:);  
%  cp(4,:) = cp(1,:); 
%  
%  cp_mc(1,:) = [grid_dim_shape*cos(theta1) grid_dim_shape*sin(theta1)]+trans_shape;
%  cp_mc(2,:) = [-grid_dim_shape*cos(theta1) -grid_dim_shape*sin(theta1)]+trans_shape; 
%  
%  cp_mc(3,:) = cp_mc(2,:);  
%  cp_mc(4,:) = cp_mc(1,:); 
%  
end
