function [Q, varargout] = ferrari_canny_3d(center_of_mass, contacts, friction_forces, normal_force)
%FERRARI_CANNY Summary of this function goes here
%   Detailed explanation goes here
dim = size(contacts, 1);
num_contacts = size(contacts, 2);
eps = 1e-3;

%Get radius from center of mass to contacts 
r = zeros(dim, num_contacts);
for i = 1:num_contacts
    r(:,i) = (contacts(:,i) - center_of_mass); 
end

%Compute Torques at Contact
R = zeros(3,3);
W = zeros(6, 0); 
torques = cell(1, num_contacts);
f_start = 1;
f_end = 1;

for i = 1:num_contacts
    %Compute r x f via Skew Symmetric Matrix
    torques{i} = zeros(size(friction_forces{i}));
    num_forces = size(friction_forces{i}, 2);
    f_end = f_start + num_forces - 1;
    
    R(1,1) = 0;       R(1,2) = -r(3,i); R(1,3) = r(2,i); 
    R(2,1) = r(3,i);  R(2,2) = 0;       R(2,3) = -r(1,i); 
    R(3,1) = -r(2,i); R(3,2) = r(1,i);  R(3,3) = 0; 
    
    W(1:3,f_start:f_end) = friction_forces{i};
    
    for j = 1:num_forces
        torques{i}(:,j) = R*friction_forces{i}(:,j);
    end
    W(4:6,f_start:f_end) = torques{i};
    f_start = f_end+1;
end

% add soft contact model
W(:,end+1) = [zeros(3,1); normal_force];

% check unique wrenches
wrenches_invalid = false;
% for i = 1:2*num_contacts
%     for j = 1:2*num_contacts
%         if i ~= j && abs(norm(W(:,i) - W(:,j))) < 1e-2
%            wrenches_invalid = true;
%            break;
%         end
%     end
% end

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

% [K, v] = convhulln(W', {'Qt', 'Qx', 'Pp', 'QJ'});
% 
% %trisurf(K,X(:,1),X(:,2),X(:,3))
% 
% %Find closest facet to origin
% min_b = 1000; 
% 
% for i=1:size(K,1)
%      for k = 1:size(K,2)
%          X(k,:) = W(:,K(i,k))';
%      end
%      b = find_dist_to_origin(X);
%      if(b < min_b)
%          min_b = b; 
%      end
% end

[min_b, in] = get_min_dist(W, eps);
Q = in; % FORCE CLOSURE

if ~in
    Q = -Q;
end
varargout{1} = true;

end

function [b] = find_dist_to_origin(X)

AB = X(2,:) - X(1,:); 
BC = X(3,:) - X(1,:); 

AB = AB/norm(AB); 
BC = BC/norm(BC); 

N = cross(AB,BC); 

b = abs(N*X(1,:)'); 

end

function [d, in] = get_min_dist(W, eps)
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
% positive or negative weights, which means it could be scaled to satisfy
% the convex hull)
[U, S, V] = svd(W);
m = size(S, 1);
sig_min = S(m,m);
in = false;
if sig_min > 0.1
    in = true;
end

d = 0;

% d = sqrt(cvx_optval); % smallest dist
% in = false;
% if d < eps  
%     in = true;
% end

% if in_alt ~= in
%    wrong = 1; 
% end

end

