#include "load_obj.hpp"

#include <cstring>
#include <iostream>
#include <stdio.h>
#include <stdlib.h> /*atof*/

// Modified from FCL test utility
void LoadOBJFile(const char* filename, std::vector<std::vector<float> >& points,
                 std::vector<std::vector<float> >& triangles)
{
  FILE* file = fopen(filename, "rb");
  if(!file) {
    std::cerr << "File does not exist" << std::endl;
    return;
  }

  bool has_normal = false;
  bool has_texture = false;
  char line_buffer[2000];
  while(fgets(line_buffer, 2000, file)) {
    char* first_token = strtok(line_buffer, "\r\n\t ");
    if(!first_token || first_token[0] == '#' || first_token[0] == 0)
      continue;

    switch(first_token[0])
    {
    case 'v':
      {
        if(first_token[1] == 'n') {
          strtok(NULL, "\t ");
          strtok(NULL, "\t ");
          strtok(NULL, "\t ");
          has_normal = true;
        }
        else if(first_token[1] == 't') {
          strtok(NULL, "\t ");
          strtok(NULL, "\t ");
          has_texture = true;
        }
        else {
          float x = (float)atof(strtok(NULL, "\t "));
          float y = (float)atof(strtok(NULL, "\t "));
          float z = (float)atof(strtok(NULL, "\t "));
          std::vector<float> p(3);
          p[0]=x;
          p[1]=y;
          p[2]=z;
          points.push_back(p);
        }
      }
      break;
    case 'f':
      {
        //fcl::Triangle tri;
        std::vector<float> tri(3);
        char* data[30];
        int n = 0;
        while((data[n] = strtok(NULL, "\t \r\n")) != NULL) {
          if(strlen(data[n]))
            n++;
        }

        for(int t = 0; t < (n - 2); ++t) {
          if((!has_texture) && (!has_normal)) {
            tri[0] = atoi(data[0]) - 1;
            tri[1] = atoi(data[1]) - 1;
            tri[2] = atoi(data[2]) - 1;
          }
          else {
            const char *v1;
            for(int i = 0; i < 3; i++) {
              // vertex ID
              if(i == 0)
                v1 = data[0];
              else
                v1 = data[t + i];
              
              tri[i] = atoi(v1) - 1;
            }
          }
          triangles.push_back(tri);
        }
      }
    }
  }
}
