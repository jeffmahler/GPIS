function data_rep = repair_object_ids(data, trials_per_object)
    num_data_points = size(data, 1);
    data_rep = data;
    obj_id = 0;
    for i = 1:trials_per_object:num_data_points
        for j = 0:(trials_per_object-1)
            data_rep(i+j,1) = obj_id;
        end
        obj_id = obj_id+1;
    end
end