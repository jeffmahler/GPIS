function loa = create_ap_loa(grasp, gripWidth)
%CREATE_AP_LOA Summary of this function goes here
%   Detailed explanation goes here

if nargin < 2
    gripWidth = 1.75;
end

% recover grasp endpoints from vector
d = size(grasp,1) / 2; % should be 2
x1 = grasp(1:d,1);
x2 = grasp(d+1:2*d,:);

% compute grasp end points from grasp center
center = x1 + (x2 - x1) / 2;
diff = (x1 - x2);
diff = diff / norm(diff);

x_start_1 = center + (gripWidth / 2) * diff;
x_start_2 = center - (gripWidth / 2) * diff;

% put in contact points list (redundant, i know)
cp1 = [x_start_1'; x_start_2'];
cp2 = [x_start_2'; x_start_1'];
loa = [cp1; cp2];
end

