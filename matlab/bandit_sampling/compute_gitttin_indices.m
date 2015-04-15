
function [] = compute_gitttin_indices(in_filename, out_filename)
T = 1500; 
 
Indices = zeros(T,T); 

if nargin < 1
    in_filename = 'matlab/bandit_sampling/indices';
end
if nargin < 2
    out_filename = 'matlab/bandit_sampling/gittins_indices';
end

load(in_filename); 

Indices(1:199,1:199) = 0.02*indices(2:200,2:200); 

for i=1:T
    for j=200:T
        Indices(i,j) = i/(i+j);
    end
end

for i = 200:T
    for j=1:T
        Indices(i,j) = i/(i+j);
    end
end
indices = Indices; 

save(out_filename,'indices'); 
end

function prob = tran_next_state(state,next_state,T)
    
    prob = 0; 
    size = [T; T]; 
    [a_s,b_s] = ind2sub(size,state); 
    [a_sp1,b_sp1] = ind2sub(size,next_state); 
    
    if(a_sp1 == a_s+1 && b_s == b_sp1)
        prob = a_s/(b_s+a_s);
    elseif(a_sp1 == a_s && b_s == b_sp1+1)
        prob = 1-a_s/(b_s+a_s);    
    else
        prob = 0; 
        
    end
    return; 
end

function prob = tran_restart(state,next_state,T,s_0)
    
    prob = 0; 
    size = [T; T]; 
    [a_s0,b_s0] = ind2sub(size,s_0); 
    [a_s,b_s] = ind2sub(size,state); 
    [a_sp1,b_sp1] = ind2sub(size,next_state); 
    
    if(a_sp1 == a_s0 && b_sp1 == b_s0)
        prob = 1;    
    else
        prob = 0;
    end
    return; 
end

function reward = r_restart(state,next_state,T)
    
    reward = 0; 
    return; 
end

function prob = r_next_state(state,next_state,T)
    
    prob = 0; 
    size = [T; T]; 
    [a_s,b_s] = ind2sub(size,state); 
    [a_sp1,b_sp1] = ind2sub(size,next_state); 
    
    if(a_sp1-a_s == 1 && b_s == b_sp1)
        reward = 1;   
    else
        reward = 0;
    end
    return; 
end