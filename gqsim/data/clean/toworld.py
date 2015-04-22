#!/usr/bin/env python3

import argparse

TEMPLATE = open('template.world.xml').read()

class NumTuple:
    def __init__(self, item):
        self.data = list(map(float, item))
    def __getitem__(self, i):
        return self.data[i]
    def __setitem__(self, i, val):
        self.data[i] = val
    def __str__(self):
        return ' '.join(map(str, self.data))

def parse_obj(infile):
    vertices = []
    faces = []
    with open(infile) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            char, *item = line.split()
            if char == 'v':
                vertex = NumTuple(item)
                vertices.append(vertex)
            elif char == 'f':
                face = NumTuple(item)
                # obj indexes from 1 to n, but gqsim indexes from 0 to n-1
                face.data = [n-1 for n in face.data]
                faces.append(face)

    # post-processing
    z_min = min(v[2] for v in vertices)
    if z_min < 0:
        for vertex in vertices: # shift so that entire object is above ground
            vertex[2] += abs(z_min)
    return vertices, faces

def write_world(infile, vertices, faces, max_size=0.05):
    name = infile.split('.obj')[0]
    outfile = name + '.world.xml'
    vertices_text = ['{}'.format(v) for v in vertices]
    faces_text = ['{}'.format(f) for f in faces]

    # re-scale so that all x/y coords are <= max_size
    xys = [max(v[0] for v in vertices), min(v[0] for v in vertices),
           max(v[1] for v in vertices), min(v[1] for v in vertices)]
    farthest = max(abs(d) for d in xys)
    scale = [str(max_size / farthest)] * 3
    with open(outfile, 'w') as f:
        f.write(TEMPLATE.format(name=name,
                                vertices=' '.join(vertices_text),
                                faces=' '.join(faces_text),
                                scale=' '.join(scale)))

def main():
    parser = argparse.ArgumentParser(
        description='Convert clean .obj files to .world.xml files.'
    )
    parser.add_argument('infile', help='.obj file')
    args = parser.parse_args()

    vertices, faces = parse_obj(args.infile)
    write_world(args.infile, vertices, faces)

if __name__ == '__main__':
    main()
