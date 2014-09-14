function val = gripper_width_constraint(x, gpModel, W)
% return the difference between the grasp points and the gripper width
% constraint W (the maximum amount a gripper can open)

d = size(x,1) / 2;
xp = [x(1:d,1)'; x(d+1:2*d,1)'];
diff = xp(1,:)' - xp(2,:)';

% compute diff
val = norm(diff) - W;
end


