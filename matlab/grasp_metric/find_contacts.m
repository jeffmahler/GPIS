function [contacts, success] = find_contacts(loas, tsdf)

n_loas = size(loas,2);
dim = size(loas{1},2);
contacts = zeros(n_loas, dim);
tsdf_dims = size(tsdf);
success = true;

for i = 1:n_loas
    cur_tsdf_val = 10.0;
    contact_found = false;
    t = 1;
    
    loa = loas{i};
    n_loa = size(loa, 1);
    
    while t <= n_loa && ~contact_found
        prev_tsdf_val = cur_tsdf_val;

        cur_point = round(loa(t,:));
        if sum(cur_point < 1) == 0 && sum(cur_point > tsdf_dims) == 0
            if size(tsdf, 3) > 1
                cur_tsdf_val = tsdf(cur_point(1), cur_point(2), cur_point(3));
            else
                cur_tsdf_val = tsdf(cur_point(2), cur_point(1));
            end
        end

        % look for sign change
        if sign(cur_tsdf_val) ~= sign(prev_tsdf_val)
            contacts(i,:) = loa(t,:);
            contact_found = true;
        end
        t = t+1;
    end
    if ~contact_found
        success = false;
    end
end

