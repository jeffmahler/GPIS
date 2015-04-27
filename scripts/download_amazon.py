import os
import sys
import json
import urllib
import urllib2

output_directory = "./amazon_picking_challenge"

# You can either set this to "all" or a list of the objects that you'd like to
# download.
objects_to_download = "all"
# objects_to_download = ["detergent", "colgate_cool_mint"]

# You can edit this list to only download certain kinds of files.
# 'rgbd' contains all of the depth maps and images from the Carmines
# 'rgb_highres' contains all of the high-res images from the Canon cameras
# 'processed' contains all of the segmented point clouds and textured meshes
# 'kinbody' contains OpenRAVE kinbodies for each object.
# See the website for more details.
#files_to_download = ["rgbd", "rgb_highres", "processed", "kinbody"]
files_to_download = ["processed"]

# Extract all files from the downloaded .tgz, and remove .tgz files.
# If false, will just download all .tgz files to output_directory
extract = True

base_url = "http://rll.berkeley.edu/amazon_picking_challenge/"
objects_url = base_url + "objects.json"

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

def fetch_objects(url):
    response = urllib2.urlopen(url)
    html = response.read()
    objects = json.loads(html)
    return objects["objects"]

def download_file(url, filename):
    u = urllib2.urlopen(url)
    f = open(filename, 'wb')
    meta = u.info()
    file_size = int(meta.getheaders("Content-Length")[0])
    print "Downloading: %s (%s MB)" % (filename, file_size/1000000.0)

    file_size_dl = 0
    block_sz = 65536
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break

        file_size_dl += len(buffer)
        f.write(buffer)
        status = r"%10d  [%3.2f%%]" % (file_size_dl/1000000.0, file_size_dl * 100. / file_size)
        status = status + chr(8)*(len(status)+1)
        print status,
    f.close()

def tgz_url(object, type):
    return base_url + "data/{object}/{type}.tgz".format(object=object,type=type)

def extract_tgz(filename, dir):
    tar_command = "tar -xzf {filename} -C {dir}".format(filename=filename,dir=dir)
    os.system(tar_command)
    os.remove(filename)

if __name__ == "__main__":

    objects = fetch_objects(objects_url)

    for object in objects:
        if objects_to_download == "all" or object in objects_to_download:
            for file_type in files_to_download:
                url = tgz_url(object, file_type)
                filename = "{path}/{object}_{file_type}.tgz".format(path=output_directory,
                                                                    object=object,
                                                                    file_type=file_type)
                download_file(url, filename)
                if extract:
                    extract_tgz(filename, output_directory)
