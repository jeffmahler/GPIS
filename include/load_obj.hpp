// File for loading OBJ files into vectors of vertices and triangles
#include <vector>

void LoadOBJFile(const char* filename, std::vector<std::vector<float> >& points,
                 std::vector<std::vector<float> >& triangles);
