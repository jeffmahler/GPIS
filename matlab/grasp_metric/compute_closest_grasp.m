function [ val_lower, val_higher ] = compute_closest_grasp( grasp_set,grasp,fric,max_r)
%COMPUTE_CLOSEST_GRASP Summary of this function goes here
%   Detailed explanation goes here

r1_p = grasp_set(:,2:3);
r2_p = grasp_set(:,4:5); 
n1_p = grasp_set(:,6:7); 
n2_p = grasp_set(:,8:9); 

Dif = zeros(size(r1_p(:,1)));

r1 = grasp(:,1:2); 
r2 = grasp(:,3:4); 
n1 = grasp(:,5:6); 
n2 = grasp(:,7:8); 

for i = 1:size(r1_p(:,1))
    sum = norm(r1_p(i,:)-r1)+norm(r2_p(i,:)-r2)+norm(n1_p(i,:)-n1)+norm(n2_p(i,:)-n2);
    Dif(i,1) = (1+fric)*(1+max_r)*sum;
end

[V, I] = min(Dif); 

Q_p= grasp_set(I,1); 

val_lower = Q_p -V; 
val_higher = Q_p+V; 

end

