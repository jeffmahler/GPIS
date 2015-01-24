import os
import urllib2

#script for downloading all objects from modelnet

pathz="http://vision.cs.princeton.edu/projects/2014/ModelNet/data/"

def getfiles(obj):
	print "starting "+obj
	f="ModelNet/"+obj
	# if not os.path.exists(f):
	# 	os.makedirs(f)
	category2read=pathz+obj+"/"
	req=urllib2.Request(category2read)
	resp=urllib2.urlopen(req)
	html=resp.read()
	i=html.find("[DIR]")+3
	i=html.find("href",i)+4
	list_folders=[]
	while i<len(html):
		temp=html.find("href=",i)+6
		if temp==5:break
		tempend=html.find("/",temp)
		folder=html[temp:tempend+1]
		list_folders.append(folder)
		i=tempend

	x=0
	for dir_file in list_folders:
		if dir_file.find(".")!=-1:continue
		# if (obj!='bowl') | (x>22):
		print "starting "+obj+"/"+dir_file
		g=f+"/"+dir_file
		# if not os.path.exists(g[0:(len(g)-1)]):
		# 	os.makedirs(g[0:(len(g)-1)])
		files=[]
		# print category2read+dir_file
		req=urllib2.Request(category2read+dir_file)
		resp=urllib2.urlopen(req)
		html=resp.read()
		
		temp=html.find(".off")
		t1=html.rfind("=",0,temp)+2
		t2=html.find(">",temp)-1
		offfile=html[t1:t2]
		lf= g+offfile
		lfopen=open(lf,"wb")
		req=urllib2.Request(category2read+dir_file+offfile)
		resp=urllib2.urlopen(req)
		html=resp.read()
		lfopen.write(html)
		lfopen.close()
		# while i<len(html):
		# 	temp=html.find("href=",i)+6
		# 	if temp==5:break
		# 	tempend=html.find(">",temp)-1
		# 	folder=html[temp:tempend]
		# 	files.append(folder)
		# 	i=tempend
		# # print files
		# for filename in files:
		# 	lf=g+filename
		# 	lfopen=open(lf,"wb")
		# 	req=urllib2.Request(category2read+dir_file+filename)
		# 	resp=urllib2.urlopen(req)
		# 	html=resp.read()
		# 	lfopen.write(html)
		# 	lfopen.close()
		x+=1
	print "done with "+obj

getfiles( 'ashtray');
getfiles( 'bar_of_soap');
getfiles( 'basket');
getfiles( 'bin');
getfiles( 'blender');
getfiles( 'bottle');
getfiles( 'bouquet');
getfiles( 'bowl');
getfiles( 'brush');
getfiles( 'bucket');
getfiles( 'butcher_knife');

getfiles( 'can');
getfiles( 'carton');
getfiles( 'coffee_cup');
getfiles( 'coffee_machine');
getfiles( 'coffee_maker');
getfiles( 'coffee_pot');
getfiles( 'decorative_platter');
getfiles( 'dish_rack');
getfiles( 'dishes');
getfiles( 'dishwasher');
getfiles( 'faucet');
getfiles( 'fire_extinguisher');
getfiles( 'flashlight');
getfiles( 'fork');
getfiles( 'fruit_bowl');
getfiles( 'frying_pan');
getfiles( 'hammer');
getfiles( 'hanger');
getfiles( 'hangers');
getfiles( 'hooks');
getfiles( 'jar');
getfiles( 'jug');
getfiles( 'kettle');
getfiles( 'kitchen_items');
getfiles( 'knife');
getfiles( 'litter_bin');
getfiles( 'mug');
getfiles( 'paintbrush');
getfiles( 'paper_cup');
getfiles( 'pan');
getfiles( 'pitcher');
getfiles( 'plate');
getfiles( 'pot');
getfiles( 'razor');
getfiles( 'ruler');
getfiles( 'saucepan');
getfiles( 'saucer');
getfiles( 'screwdriver');
getfiles( 'shovel');
getfiles( 'soap');
getfiles( 'soap_bottle');
getfiles( 'spoon');
getfiles( 'tea_pot');
getfiles( 'toaster');
getfiles( 'torch');
getfiles( 'wine_bottle');
getfiles( 'wine_rack');
getfiles( 'wine_glass');
getfiles( 'wrench');
getfiles( 'wire');

print "finito"