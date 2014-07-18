%Grasp Parameters
%Grasp points along line of action 

close all;
 
cm = [12 12]; 
fric_coef = 0.5;
cone_angle = atan(fric_coef);
num_contacts = 2; 

num_grasps = 1;

grid_size = sqrt(size(allPoints,1)); 

for i =1:num_grasps
    
    [cp1,cp2] = get_random_grasp(grid_size);
    
    
    cp = [cp1; cp2];
    tic 
    q = MC_sample( gpModel,allPoints,cone_angle,cp,num_contacts,cm)
    toc

    
    
end