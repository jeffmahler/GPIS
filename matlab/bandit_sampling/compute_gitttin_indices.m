discount = 0.99; 
T = 1500; 
indices = zeros(T,T); 
i = T; 

while(i ~= 0)
    j = T; 
    while(j ~= 0)
        if(i == T && j == T)
            indices(i,j) = i/(i+j);
        elseif(j == T)
             indices(i,j) = i/(i+j) + discount*((i)/(i+j)*indices(i+1,j)); 
        elseif(i == T)
             indices(i,j) = i/(i+j) + discount*(j/(i+j)*indices(i,j+1));
        else 
            indices(i,j) = i/(i+j) + discount*((i)/(i+j)*indices(i+1,j)+j/(i+j)*indices(i,j+1));
          
            
        end
        j = j-1; 
    end 
    i = i-1; 
end
        
save('matlab/bandit_sampling/gittins_indices','indices'); 