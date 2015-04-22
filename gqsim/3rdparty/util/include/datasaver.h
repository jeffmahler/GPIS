//================================================================================
//         DATA SAVER FOR VARYING SIZE DATA
// 
//                                                               junggon@gmail.com
//================================================================================

#ifndef _DATA_SAVER_
#define _DATA_SAVER_

#include <fstream>
#include <vector>
#include <list>

template <typename TYPE> class DataSaverVaryingSize
{
public:
	// data to be saved
	std::vector<TYPE> *m_ptr_data;	// pointer to a vector containing TYPE data to be saved (fixed size vector)
	std::vector<int> *m_ptr_indices;	// pointer to a vector containing the data indices to be saved (varying size vector)

	// memory for saving data bundle
	std::list< std::vector<TYPE> > m_list_data;	// memory where the bundle of data to be saved
	std::list< std::vector<int> > m_list_indices;	// memory where the bundle of data indices to be saved

	// constructors/destructor
	DataSaverVaryingSize() : m_ptr_data(NULL), m_ptr_indices(NULL) {}
	DataSaverVaryingSize(std::vector<TYPE> *ptr_data, std::vector<int> *ptr_indices) : m_ptr_data(ptr_data), m_ptr_indices(ptr_indices) {}
	~DataSaverVaryingSize() {}

	void set_data_to_be_saved(std::vector<TYPE> *ptr_data, std::vector<int> *ptr_indices) 
	{ 
		m_ptr_data = ptr_data; 
		m_ptr_indices = ptr_indices; 
		m_list_data.clear(); 
		m_list_indices.clear(); 
	}

	void clear_memory()
	{
		m_list_data.clear(); 
		m_list_indices.clear(); 
	}

	bool write_data_into_memory()
	{
		if ( !m_ptr_data || !m_ptr_indices ) return false;
		std::vector<TYPE> data(m_ptr_indices->size());
		for (size_t i=0; i<data.size(); ++i) { 
			if ( m_ptr_indices->at(i) < 0 || m_ptr_indices->at(i) >= (int)m_ptr_data->size() ) return false;
			data[i] = m_ptr_data->at(m_ptr_indices->at(i)); 
		}
		m_list_data.push_back(data);
		m_list_indices.push_back(*m_ptr_indices);
		return true;
	}

	void set_data_all(TYPE d)
	{
		for (size_t i=0; i<m_ptr_data->size(); ++i) {
			m_ptr_data->at(i) = d;
		}
	}

	bool read_data_from_memory(int idx)
	{
		if ( !m_ptr_data || !m_ptr_indices ) return false;
		typename std::list< std::vector<TYPE> >::iterator iter_list_data = m_list_data.begin(); std::advance(iter_list_data, idx);
		std::list< std::vector<int> >::iterator iter_list_indices = m_list_indices.begin(); std::advance(iter_list_indices, idx);
		std::vector<TYPE> &data = *iter_list_data;
		std::vector<int> &indices = *iter_list_indices;
		for (size_t i=0; i<indices.size(); ++i) {
			if ( indices[i] < 0 || indices[i] >= (int)m_ptr_data->size() ) return false;
			m_ptr_data->at(indices[i]) = data[i];
		}
		*m_ptr_indices = indices;
		return true;
	}

	void truncate_data_on_memory(int idx)
	{
		if ( idx < 0 || idx >= (int)m_list_data.size() ) return;
		typename std::list< std::vector<TYPE> >::iterator iter_data = m_list_data.begin();
		std::list< std::vector<int> >::iterator iter_indices = m_list_indices.begin();
		std::advance(iter_data, idx);
		std::advance(iter_indices, idx);
		m_list_data.erase(iter_data, m_list_data.end());
		m_list_indices.erase(iter_indices, m_list_indices.end());
	}

	//---------------------- file format (binary) -----------------------------
	// n                                                      // n = size of the data bundle
	// m[0] idx[0][0] ... idx[0][m[0]-1]                      // m[0] = size of the first indices, idx[0] = the first indices
	// m[0] data[0][0] ... data[0][m[0]-1]                    // m[0] = size of the first data, data[0] = the first data
	// ...
	// m[n-1] idx[n-1][0] ... idx[n-1][m[n-1]-1]              // m[n-1] = size of the last indices, idx[n-1] = the last indices
	// m[n-1] data[n-1][0] ... data[n-1][m[n-1]-1]            // m[n-1] = size of the last data, data[n-1] = the last data
	//-------------------------------------------------------------------------
	bool save_data_on_memory_into_file(std::ofstream *pfout)
	{
		if ( !pfout ) return false;
		if ( m_list_data.size() != m_list_indices.size() ) return false;
		int i;
		std::list< std::vector<int> >::iterator iter_indices;
		typename std::list< std::vector<TYPE> >::iterator iter_data;
		int n = (int)m_list_data.size();
		pfout->write((char *)&n, sizeof(int));
		int m;
		for (i=0, iter_indices = m_list_indices.begin(), iter_data = m_list_data.begin(); i<n; ++i, ++iter_indices, ++iter_data) {
			m = (int)(*iter_indices).size();
			pfout->write((char *)&m, sizeof(int));
			if ( m > 0 ) {
				pfout->write((char *)&((*iter_indices)[0]), m*sizeof(int));
			}
			m = (int)(*iter_data).size();
			pfout->write((char *)&m, sizeof(int));
			if ( m > 0 ) {
				pfout->write((char *)&((*iter_data)[0]), m*sizeof(TYPE));
			}
		}
		return true;
	}

	bool load_data_on_memory_from_file(std::ifstream *pfin)
	{
		if ( !pfin ) return false;
		int i;
		std::list< std::vector<int> >::iterator iter_indices;
		typename std::list< std::vector<TYPE> >::iterator iter_data;
		int n;
		pfin->read((char *)&n, sizeof(int));
		m_list_indices.resize(n);
		m_list_data.resize(n);
		int m;
		for (i=0, iter_indices = m_list_indices.begin(), iter_data = m_list_data.begin(); i<n; ++i, ++iter_indices, ++iter_data) {
			pfin->read((char *)&m, sizeof(int));
			(*iter_indices).resize(m);
			if ( m > 0 ) {
				pfin->read((char *)&((*iter_indices)[0]), m*sizeof(int));
			}
			pfin->read((char *)&m, sizeof(int));
			(*iter_data).resize(m);
			if ( m > 0 ) {
				pfin->read((char *)&((*iter_data)[0]), m*sizeof(TYPE));
			}
		}

		return true;
	}

	bool load_data_on_memory_from_file_app(std::ifstream *pfin)
	{
		if ( !pfin ) return false;
		int i;
		int n;
		std::list< std::vector<int> >::iterator iter_indices;
		typename std::list< std::vector<TYPE> >::iterator iter_data;
		pfin->read((char *)&n, sizeof(int));
		std::list< std::vector<int> > list_indices(n);
		std::list< std::vector<TYPE> > list_data(n);
		int m;
		for (i=0, iter_indices = list_indices.begin(), iter_data = list_data.begin(); i<n; ++i, ++iter_indices, ++iter_data) {
			pfin->read((char *)&m, sizeof(int));
			(*iter_indices).resize(m);
			if ( m > 0 ) {
				pfin->read((char *)&((*iter_indices)[0]), m*sizeof(int));
			}
			pfin->read((char *)&m, sizeof(int));
			(*iter_data).resize(m);
			if ( m > 0 ) {
				pfin->read((char *)&((*iter_data)[0]), m*sizeof(TYPE));
			}
		}

		// append the indices and data
		m_list_indices.splice(m_list_indices.end(), list_indices);
		m_list_data.splice(m_list_data.end(), list_data);

		return true;
	}

	void _write_indices(std::ofstream *pfout, std::vector<int> &indices)
	{
		int i;
		std::vector<int>::iterator iter;
		int *buffer = new int[indices.size()];
		for (i=0, iter = indices.begin(); iter != indices.end(); ++i, iter++) {
			buffer[i] = *iter;
		}
		pfout->write((char *)buffer, indices.size()*sizeof(int));
		delete [] buffer;
	}

	void _write_data(std::ofstream *pfout, std::vector<TYPE> &data)
	{
		int i;
		typename std::vector<TYPE>::iterator iter;
		TYPE *buffer = new TYPE[data.size()];
		for (i=0, iter = data.begin(); iter != data.end(); ++i, iter++) {
			buffer[i] = *iter;
		}
		pfout->write((char *)buffer, data.size()*sizeof(TYPE));
		delete [] buffer;
	}

	void _read_indices(std::ifstream *pfin, std::vector<int> &indices)
	{
		int i;
		std::vector<int>::iterator iter;
		int *buffer = new int[indices.size()];
		pfin->read((char *)buffer, indices.size()*sizeof(int));
		for (i=0, iter = indices.begin(); iter != indices.end(); ++i, iter++) {
			*iter = buffer[i];
		}
		delete [] buffer;
	}

	void _read_data(std::ifstream *pfin, std::vector<TYPE> &data)
	{
		int i;
		typename std::vector<TYPE>::iterator iter;
		TYPE *buffer = new TYPE[data.size()];
		pfin->read((char *)buffer, data.size()*sizeof(TYPE));
		for (i=0, iter = data.begin(); iter != data.end(); ++i, iter++) {
			*iter = buffer[i];
		}
		delete [] buffer;
	}
};


#endif

