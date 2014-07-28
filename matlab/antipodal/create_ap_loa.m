function loa = create_ap_loa(grasp, lengthScale)
%CREATE_AP_LOA Summary of this function goes here
%   Detailed explanation goes here

if nargin < 2
    lengthScale = 1.75;
end

d = size(grasp,1) / 2; % should be 2

x1 = grasp(1:d,1);
x2 = grasp(d+1:2*d,:);

x_start_1 = x2 + lengthScale*(x1 - x2);
x_start_2 = x1 + lengthScale*(x2 - x1);

cp1 = [x_start_1'; x2'];
cp2 = [x_start_2'; x1'];
loa = [cp1; cp2];
end

