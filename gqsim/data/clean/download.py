import os
import re

import toworld

SHAPE_ROOT = '/mnt/terastation/shape_data'

def scp(remote, local, force=False):
    if not os.path.exists(local) or force:
        command = 'scp rll:{} {}'.format(remote, local)
        print(command)
        os.system(command)

def convert(local, force=False):
    if not os.path.exists(local.replace('.obj', '.world.xml')) or force:
        print('converting', local)
        vertices, faces = toworld.parse_obj(local)
        toworld.write_world(local, vertices, faces)

ycb_pattern = r'{}/YCB/(?P<name>.*?)/'.format(SHAPE_ROOT)
def ycb_handler(path):
    fname = re.match(ycb_pattern, path).group('name') + '.obj'
    local_path = 'YCB/{}'.format(fname)

    scp(path, local_path)
    convert(local_path)

cat50_pattern = \
    r'{}/Cat50_ModelDatabase/(?P<name>.*?)/(?P<hash>.*?)_clean.obj'.format(SHAPE_ROOT)
def cat50_handler(path):
    m = re.match(cat50_pattern, path)
    fname = m.group('name') + '-' + m.group('hash') + '.obj'
    local_path = 'Cat50_ModelDatabase/{}'.format(fname)

    scp(path, local_path)
    convert(local_path)

def main():
    handlers = {
        'YCB': ycb_handler,
        'Cat50_ModelDatabase': cat50_handler,
    }

    for dataset, handler in handlers.items():
        with open(os.path.join(dataset, 'index')) as f:
            for fname in f:
                handler(fname.strip())

if __name__ == '__main__':
    main()
