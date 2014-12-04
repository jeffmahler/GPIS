function centers = octree_bin_centers(oct)

centers = (oct.BinBoundaries(:,1:3) + oct.BinBoundaries(:,4:6)) / 2;

end

