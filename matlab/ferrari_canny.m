function [ Q ] = ferrari_canny( center_of_mass,p,f )
%FERRARI_CANNY Summary of this function goes here
%   Detailed explanation goes here

num_contacts = size(p,2); 

%Get radius from center of mass to contacts 
for i=1:num_contacts
    r(:,i) = p(:,i) - center_of_mass; 
end


%Compute Torques at Contact
 R = zeros(3,3); 
 index = 1;
for i=1:num_contacts
    %Compute r x f via Skew Symmetric Matrix
   
    R(1,1) = 0;       R(1,2) = -r(3,i); R(1,3) = -r(2,i); 
    R(2,1) = r(3,i);  R(2,2) = 0;       R(2,3) = -r(1,i); 
    R(3,1) = -r(2,i); R(3,2) = r(1,i);  R(3,3) = 0; 
    
    t(:,index) = R*f(:,index);
  %  t(:,index) = t(:,index); 
    index = index+1;
    t(:,index) = R*f(:,index);
  %  t(:,index) = t(:,index)/norm(t(:,index),2); 
    index = index+1;
end

%Compose Wrenches 
W = zeros(3,2*num_contacts); 

W(1:2,:) = f(1:2,:); 
W(3,:) = t(3,:); 

%TODO look up plane from triangle calculation 
%TODO a*x=b has to have ||a||=1

[K, v] = convhulln(W');

%trisurf(K,X(:,1),X(:,2),X(:,3))

%Find closest facet to origin
min_b = -1000; 

for i=1:size(K,1)
     for k = 1:3
         X(k,:) = W(:,K(i,k))';
     end
     b = find_b(X);
     if(min_b < b)
         min_b = b; 
     end
end

Q = min_b;

end

function [b] = find_b(X)

AB = X(1,:) - X(2,:); 
BC = X(2,:) - X(3,:); 

AB = AB/norm(AB,2); 
BC = BC/norm(BC,2); 

N = cross(AB,BC); 

b = N*X(1,:)'; 

end

