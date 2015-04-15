function diff = grasp_dist(grasp1, grasp2)

beta = 0.1;
d = norm(grasp2.t_grasp_obj - grasp1.t_grasp_obj);
a = grasp1.dir * grasp2.dir' / (norm(grasp1.dir) * norm(grasp2.dir));
diff = d + beta * abs(a);

end

