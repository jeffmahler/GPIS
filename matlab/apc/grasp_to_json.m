function grasp_json = grasp_to_json(grasp)
%GRASP_TO_JSON
num_rots = size(grasp.R_g_obj_list, 2);
grasp_json = {};
for i = 1:num_rots
    grasp_json{i} = struct();
    grasp_json{i}.flag = false;
    grasp_json{i}.gripper_width = 0.075;
    grasp_json{i}.gripper_pose = struct();
    
    grasp_json{i}.gripper_pose.position = struct();
    grasp_json{i}.gripper_pose.position.x = grasp.t_g_obj(1);
    grasp_json{i}.gripper_pose.position.y = grasp.t_g_obj(2);
    grasp_json{i}.gripper_pose.position.z = grasp.t_g_obj(3);
    
    q = dcm2quat(grasp.R_g_obj_list{i});
    grasp_json{i}.gripper_pose.orientation = struct();
    grasp_json{i}.gripper_pose.orientation.x = q(1);
    grasp_json{i}.gripper_pose.orientation.y = q(2);
    grasp_json{i}.gripper_pose.orientation.z = q(3);
    grasp_json{i}.gripper_pose.orientation.w = q(4);
end

grasp_json = horzcat(grasp_json{:});

