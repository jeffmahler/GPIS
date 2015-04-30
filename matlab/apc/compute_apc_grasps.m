function all_grasp_results = compute_apc_grasps(config)
%COMPUTE_APC_GRASPS Loop through all objects in the input directory
% and compute parallel-jaw grasps for this using

% loop through all files in the root directory
root_dir = config.root_dir;
files = dir(root_dir);
num_files = size(files, 1);
all_grasp_results = {};
ind = 1;

for i = 26:num_files
    cur_file = files(i);
    if cur_file.isdir && strcmp(cur_file.name, '.') == 0 && ...
        strcmp(cur_file.name, '..') == 0
    
        % get filenames
        object_name = cur_file.name;
        fprintf('Computing grasps for %s\n', object_name);

        % get grass
        grasp_results = compute_mesh_grasps(object_name, config);
        all_grasp_results{ind} = grasp_results;
        ind = ind + 1;

%         if ind > 1
%             break;
%         end
    end
end

end

