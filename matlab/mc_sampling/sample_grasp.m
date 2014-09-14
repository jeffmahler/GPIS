function graspPoints = sample_grasp(points, graspSigma)
% Sample grasp points to model uncertainty in the approach direction

if nargin < 2
   graspSigma = 0; 
end

n = size(points,1);
graspPoints = points;

% sample the endpoints of the grasp
if graspSigma > 0
    for i = 1:(n/2)
       graspPoints(i,:) = normrnd(points(i,:), graspSigma); 
    end
    
    % these are constrained to be equal
    graspPoints(3,:) = graspPoints(2,:);
    graspPoints(4,:) = graspPoints(1,:);
end

end

