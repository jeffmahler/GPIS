#ifndef _FINGER_PRINT_
#define _FINGER_PRINT_

#include <vector>
#include <list>
#include <fstream>

struct FingerPrint
{
	int _hit_cnt_ref;								// reference value for the hit count
	std::vector< std::vector<int> > _face_hit_cnt;	// _face_hit_cnt[i][j] = number of contact hits at the j-th face of the i-th surface
};

class FingerPrintHistory
{
public:

	std::list<FingerPrint> _fingerprints;			// list of fingerprints

public:
	FingerPrintHistory() {}
	~FingerPrintHistory() {}

	size_t size() const { return _fingerprints.size(); }

	void add(FingerPrint fp) { _fingerprints.push_back(fp); }
	
	void save(const char *filepath) 
	{
		std::ofstream fout(filepath);
		fout << _fingerprints.size() << std::endl;
		fout << std::endl;
		for ( std::list<FingerPrint>::iterator iter = _fingerprints.begin(); iter != _fingerprints.end(); iter++) {
			fout << iter->_hit_cnt_ref << std::endl;
			fout << iter->_face_hit_cnt.size() << std::endl;
			for (size_t j=0; j<iter->_face_hit_cnt.size(); j++) {
				fout << iter->_face_hit_cnt[j].size() << std::endl;
				for (size_t k=0; k<iter->_face_hit_cnt[j].size(); k++) {
					fout << iter->_face_hit_cnt[j][k] << " ";
				}
				fout << std::endl;
			}
			fout << std::endl;
		}
		fout.close();
	}
	
	bool load(const char *filepath)
	{
		std::ifstream fin(filepath);
		int n, ns, nf;
		fin >> n;
		_fingerprints.resize(n);
		for ( std::list<FingerPrint>::iterator iter = _fingerprints.begin(); iter != _fingerprints.end(); iter++) {
			fin >> iter->_hit_cnt_ref;
			fin >> ns;
			iter->_face_hit_cnt.resize(ns);
			for (size_t j=0; j<ns; j++) {
				fin >> nf;
				iter->_face_hit_cnt[j].resize(nf);
				for (size_t k=0; k<nf; k++) {
					fin >> iter->_face_hit_cnt[j][k];
				}
			}
		}
		fin.close();
		return true;
	}
	
	FingerPrint & getFingerPrint(int idx)
	{
		std::list<FingerPrint>::iterator iter = _fingerprints.begin();
		std::advance(iter, idx);
		return *iter;
	}
};

#endif

