# -*- coding: utf-8 -*-

# pulls all files from specified categories and pages in archive3d
import urllib 
import urllib2
import os 
import os.path
import zipfile
import IPython

base_url = "http://archive3d.net/"

#unarchives a file and places its contents in destination
def unzip(filename, destination):
    with zipfile.ZipFile(filename) as zf:
        for member in zf.infolist():
            words, path = member.filename.split('/'), destination
            for word in words[:-1]:
                drive, word = os.path.splitdrive(word)
                head, word = os.path.split(word)
                if word in (os.curdir, os.pardir, ''): continue
                path = os.path.join(path, word)
            zf.extract(member, path)

def download_shapes_from(category, num, start_page, end_page):
	print "starting download: " + category

	#loop iterates over the 3 categories of shapes to download from
	url, folder, sub_folder_list = base_url + "?category=" + str(num), "Archive3D/" + category, []
	
	#page url generation
	iter_list = [''] + ['&page=' + str(1+24*x) for x in range(start_page-1, end_page)]

	#loop iterates over the pages containing objects in a given category
	for addend in iter_list:
		url += addend
		request = urllib2.Request(url)
		response = urllib2.urlopen(request)
		if not os.path.exists(folder):
			os.makedirs(folder)
		html_code = response.read()

		counter = 1
		
		#loop iterates over the objects on a given page, downloading, unarchiving, and placing each one in the appropriate location
		counters = {}		
		while True:
			index = html_code.find('<a href="?a=download&amp;id=')
			if index == -1: break
			url_2 = base_url + html_code[index+9:index+36]
			url_2 = url_2.replace('amp;', '')
			sub_request = urllib2.Request(url_2)
			sub_response = urllib2.urlopen(sub_request)
			html_code, html_code_2 = html_code[html_code.find('title="Download')+16:], sub_response.read()
			obj = html_code[:html_code.find("3")-1]
			obj = obj.lower().strip() # normalize names by lowercase
			sub_folder = folder + "/" + obj
			print "downloading object: " + obj.lower()
			IPython.embed()
			if not os.path.exists(sub_folder):
				os.makedirs(sub_folder)
			if obj in counters.keys():
				counters[obj] += 1
				counter = counters[obj] + 1
			else:
				counter, counters[obj] = 1, 1
				IPython.embed()
			download_url, dir_name = url_2.replace('id', 'do=get&id'), obj.lower() + "_" + str(counter)
			urllib.urlretrieve(download_url, "temp")
			os.makedirs(sub_folder + "/" + dir_name)
			unzip("temp", sub_folder + "/" + dir_name)

download_shapes_from("Kitchen Ware", 429, 1, 2)

print "download complete."

