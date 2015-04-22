//================================================================================
//         UTILITY TOOLS FOR XML PARSING
// 
//                                                               junggon@gmail.com
//================================================================================

#ifndef _XML_PARSER_
#define _XML_PARSER_

#include <vector>
#include <string>
#include "tinyxml.h"
#include "gear.h"

class RigidObject;
class RigidBody;
class GJoint;

class xmlElementProperty {
public:
	xmlElementProperty() : pxmlelement(NULL), row(0), col(0) {}
	xmlElementProperty(const char *k, const char *v, int r, int c) : key(k), value(v), pxmlelement(NULL), row(r), col(c) {}
	xmlElementProperty(const char *k, TiXmlElement *p, int r, int c) : key(k), value(""), pxmlelement(p), row(r), col(c) {}
	~xmlElementProperty() {}
public:
	std::string key, value;
	TiXmlElement *pxmlelement;
	int row, col;
};

class xmlNumberArray {
public:
	xmlNumberArray() {}
	xmlNumberArray(std::string str) {
		const char *delimiters = " ,\t\n\r";
		char *str2 = new char [str.length()+1]; strcpy(str2, str.c_str());
		char *tok = strtok(str2, delimiters);
		while (tok != NULL) {
			x.push_back(atof(tok));
			tok = strtok(NULL, delimiters);
		}
		delete [] str2;
	}
	xmlNumberArray(std::string str, int nmax) {
		x.reserve(nmax);
		const char *delimiters = " ,\t\n\r";
		char *str2 = new char [str.length()+1]; strcpy(str2, str.c_str());
		char *tok = strtok(str2, delimiters);
		int cnt=0;
		while (tok != NULL) {
			if ( cnt >= nmax ) 
				break;
			x.push_back(atof(tok)); cnt++;
			tok = strtok(NULL, delimiters);
		}
		delete [] str2;
	}
	~xmlNumberArray() {}

	size_t size() const { return x.size(); }
	Vec3 get_Vec3() { return Vec3((gReal)x[0], (gReal)x[1], (gReal)x[2]); }
	SO3 get_SO3() { return SO3((gReal)x[0], (gReal)x[1], (gReal)x[2], (gReal)x[3], (gReal)x[4], (gReal)x[5], (gReal)x[6], (gReal)x[7], (gReal)x[8]); }
	SE3 get_SE3() { return SE3((gReal)x[0], (gReal)x[1], (gReal)x[2], (gReal)x[4], (gReal)x[5], (gReal)x[6], (gReal)x[8], (gReal)x[9], (gReal)x[10], (gReal)x[12], (gReal)x[13], (gReal)x[14]); }
public:
	std::vector<double> x;
};

class xmlNumberArrayInt {
public:
	xmlNumberArrayInt() {}
	xmlNumberArrayInt(std::string str) {
		const char *delimiters = " ,\t\n\r";
		char *str2 = new char [str.length()+1]; strcpy(str2, str.c_str());
		char *tok = strtok(str2, delimiters);
		while (tok != NULL) {
			x.push_back(atoi(tok));
			tok = strtok(NULL, delimiters);
		}
		delete [] str2;
	}
	~xmlNumberArrayInt() {}

	size_t size() const { return x.size(); }
public:
	std::vector<int> x;
};

class xmlStringArray {
public:
	xmlStringArray() {}
	xmlStringArray(std::string str) {
		const char *delimiters = " ,\t\n\r";
		char *str2 = new char [str.length()+1]; strcpy(str2, str.c_str());
		char *tok = strtok(str2, delimiters);
		while (tok != NULL) {
			x.push_back(std::string(tok));
			tok = strtok(NULL, delimiters);
		}
		delete [] str2;
	}
	~xmlStringArray() {}

	size_t size() const { return x.size(); }
public:
	std::vector< std::string > x;
};

// parse xml element for object and set the object's properties (if failed, return false)
bool xmlParseObject(TiXmlElement *pelement, RigidObject* pobject, std::string prefix = std::string("")); 

// parse xml element for body, create a rigid object, set the object's properties, and return the object pointer (if failed, return NULL)
RigidObject* xmlParseObject(TiXmlElement *pelement, std::string prefix = std::string("")); 

// parse xml element for object geometry
bool xmlParseObjectGeometry(TiXmlElement *pelement, RigidObject* pobject);

// parse xml element for body and set the body's properties (if failed, return false)
bool xmlParseBody(TiXmlElement *pelement, RigidBody* pbody, std::string prefix = std::string("")); 

// parse xml element for body, create a rigid body, set the body's properties, and return the body pointer (if failed, return NULL)
RigidBody* xmlParseBody(TiXmlElement *pelement, std::string prefix = std::string("")); 

// parse xml element for joint, create a joint, set the joint's properties, and return the joint pointer (if failed, return NULL)
GJoint* xmlParseJoint(TiXmlElement *pelement, std::vector<RigidBody*> pbodies, std::string prefix = std::string("")); 

// parse xml element for body geometry
bool xmlParseBodyGeometry(TiXmlElement *pelement, RigidBody* pbody);

// scan xml element properties (case insensitive -- all text will be converted to lower case)
void xmlScanElementProperties(TiXmlElement *pelement, std::vector<xmlElementProperty> &properties);

// load xml keywords
void xmlLoadKeywords(const char *filepath);
void xmlLoadKeywords(const std::vector< std::vector< std::string > > &keywords);
void xmlPrintKeywords();


#endif

