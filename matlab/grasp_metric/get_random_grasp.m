function [ cp cp_mc] = get_random_grasp(grid_dim,grid_dim_shape,random)
%Returns a random grasp specified by a line of action with start and end
%points 

if nargin < 3
    random = 0;
end

%TODO: Support more than 2 contact points
 
theta1 = rand()*pi; 
theta2 = rand()*pi+pi;

trans = grid_dim/2;
trans_shape = grid_dim_shape/2;

grid_dim = grid_dim/2;
grid_dim_shape = grid_dim_shape/2;

 if(random) 
     x_trans = rand();
     y_trans = rand(); 
 else
     x_trans = 0; 
     y_trans = 0;
 end

 cp(1,:) = [cos(theta1)+x_trans sin(theta1)+y_trans]; 
 cp(2,:) = [-cos(theta1)+x_trans -sin(theta1)+y_trans]; 
    
 cp(1,:) = grid_dim*cp(1,:)/norm(cp(1,:))+trans;
 cp(2,:) = grid_dim*cp(2,:)/norm(cp(2,:))+trans;

 cp(3,:) = cp(2,:);  
 cp(4,:) = cp(1,:); 
 
 cp_mc(1,:) = [grid_dim_shape*cos(theta1) grid_dim_shape*sin(theta1)]+trans_shape;
 cp_mc(2,:) = [-grid_dim_shape*cos(theta1) -grid_dim_shape*sin(theta1)]+trans_shape; 
 
 cp_mc(3,:) = cp_mc(2,:);  
 cp_mc(4,:) = cp_mc(1,:); 
 
end

