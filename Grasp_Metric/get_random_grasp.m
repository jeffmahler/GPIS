function [ cp1,cp2 ] = get_random_grasp(grid_size )
%Returns a random grasp specified by a line of action with start and end
%points 

%TODO: Support more than 2 contact points
 
 j = randi(grid_size); 
 k = randi(grid_size); 
 cp1(1,:) = [j 25]; 
 cp1(2,:) = [k 1]; 

 j = randi(grid_size); 
 k = randi(grid_size); 
 cp2(1,:) = [j 1]; 
 cp2(2,:) = [k 25]; 
 
 
end

