function [loa] = compute_loa(grip_point)
%Calculate Line of Action given start and end point

    step_size = 0.3; 

    start_point = grip_point(1,:); 
    end_p = grip_point(2,:); 

    grad = end_p-start_point; 
    end_time = norm(grad, 2);
    grad = grad/end_time; 
    i=1; 
    time = 0;

    while(time < end_time)
        point = start_point + grad*time;
        loa(i,:) = point;
        time = time + step_size; 
        i = i + 1;
    end
    size(loa)
end
