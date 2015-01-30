import os

#script for converting all our .off objects in the toppath directory and all its children
#directories iteratively

toppath="/mnt/terastation/shape_data/ModelNet/"
dirnames=[toppath]
#basiccommand="meshconv -c obj -tri "


def offobj(path):
	vertices_=[]
	triangles_=[]
	filez = open(path,'r')
	firstline=True
	for line in filez:
		points=line.split()
		if len(points)==0 or points[0].startswith('#') or len(points)==1:
			pass
		elif len(points)==3:
			if firstline:
				firstline=False
			else:
				x,y,z=float(points[0]),float(points[1]),float(points[2])
				vertices_.append([x,y,z])
		else:
			p1,p2,p3=points[1],points[2],points[3]
			triangles_.append([p1,p2,p3])
	filez.close()
	filecontent="#Obj File:\n"
	for vertex_ in vertices_:
		filecontent+=("v "+str(vertex_[0])+" "+str(vertex_[1])+" "+str(vertex_[2])+"\n")
	for face_ in triangles_:
		filecontent+=("f "+str(face_[0])+" "+str(face_[1])+" "+str(face_[2])+"\n")
	filez=open(path[:-4]+".obj","w")
	filez.write(filecontent)
	filez.close()

while dirnames != []:
	temp=dirnames.pop()
	# print "in " +temp+" directory"
	templist=os.listdir(temp)
	tempstring=""
	for tempy in templist:
		tempstring+=tempy
	for name in templist:
		if (name.find("Utility Source Code")!=-1) | (name.find("Documentation")!=-1):
			continue
		if (os.path.isfile(temp+name)):
			print name
			if tempstring.find(".obj")==-1 and name.find(".off")!=-1:
				offobj(temp+name)
				
			else:
				#thiscommand=basiccomand+temp+name
				pass
				#command call
		else:
			dirnames.append(temp+name+"/")