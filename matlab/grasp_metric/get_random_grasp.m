function [ cp] = get_random_grasp(grid_dim)
%Returns a random grasp specified by a line of action with start and end
%points 

%TODO: Support more than 2 contact points
 
theta1 = rand()*pi; 

theta2 = rand()*pi+pi;

trans = grid_dim/2;

grid_dim = grid_dim/2;

 cp(1,:) = [grid_dim*cos(theta1) grid_dim*sin(theta1)]+trans; 
 cp(2,:) = [-grid_dim*cos(theta1) -grid_dim*sin(theta1)]+trans; 


 cp(3,:) = [grid_dim*cos(theta2) grid_dim*sin(theta2)]+trans; 
 cp(4,:) = [-grid_dim*cos(theta2) -grid_dim*sin(theta2)]+trans;
 
 
end

