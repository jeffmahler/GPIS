function [Q, varargout] = ferrari_canny( center_of_mass,p,f )
%FERRARI_CANNY Summary of this function goes here
%   Detailed explanation goes here

num_contacts = size(p,2);
eps = 1e-5;

%Get radius from center of mass to contacts 
for i=1:num_contacts
    r(:,i) = p(:,i) - center_of_mass; 
end


%Compute Torques at Contact
 R = zeros(3,3); 
 index = 1;
for i=1:num_contacts
    %Compute r x f via Skew Symmetric Matrix
   
    R(1,1) = 0;       R(1,2) = -r(3,i); R(1,3) = r(2,i); 
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

% check unique wrenches
wrenches_invalid = false;
for i = 1:2*num_contacts
    for j = 1:2*num_contacts
        if i ~= j && abs(norm(W(:,i) - W(:,j))) < 1e-2
           wrenches_invalid = true;
           break;
        end
    end
end

% check NaNs
if sum(sum(isnan(W))) > 0
    wrenches_invalid = true;
end

% check rank
[U, S, V] = svd(W);
if S(3,3) < 1e-2
    wrenches_invalid = true;
end



%TODO look up plane from triangle calculation 
%TODO a*x=b has to have ||a||=1

if wrenches_invalid
    Q = -1.0;
    varargout{1} = false;
    return;
end

[K, v] = convhulln(W', {'Qt', 'Pp', 'QJ'});

%trisurf(K,X(:,1),X(:,2),X(:,3))

%Find closest facet to origin
min_b = 1000; 

for i=1:size(K,1)
     for k = 1:3
         X(k,:) = W(:,K(i,k))';
     end
     b = find_b(X);
     if(b < min_b)
         min_b = b; 
     end
end

Q = min_b;

in = check_zero_inside(W, eps);
if ~in
    Q = -Q;
end
varargout{1} = true;

end

function [b] = find_b(X)

AB = X(2,:) - X(1,:); 
BC = X(3,:) - X(1,:); 

AB = AB/norm(AB); 
BC = BC/norm(BC); 

N = cross(AB,BC); 

b = abs(N*X(1,:)'); 

end

function [in] = check_zero_inside(W, eps)
n = size(W,2);

% LP to check whether 0 is in the convex hull of the wrenches
% cvx_begin quiet
%     variable z(n);
%     minimize( z' * (W' * W) * z )
%     subject to
%         z >= 0;
%         sum(z) == 1;
% cvx_end

% alternative solution method (see if null vector of W^T W can have all
% positive or negative weights)
[U, S, V] = svd(W'*W);
nullVec = U(:,n);
numPositive = sum(nullVec > 0);
in = false;
if numPositive == size(nullVec,1) || numPositive == 0
    in = true;
end
    
%cvx_optval
% in_alt = false;
% if cvx_optval < eps
%     in_alt = true;
% end
% 
% if in_alt ~= in
%    wrong = 1; 
% end

end

