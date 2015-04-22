#include <fstream>
#include <iomanip>
#include <vector>
#include <vector>
#include <algorithm>
#include "simuldata.h"

using namespace std;

static vector<double> _data_double;
static vector<float> _data_float;
static vector<int> _data_int;
static vector<bool> _data_bool;

bool SimulData::set_data_double(vector<double> &data_double_) 
{
	if ( data_double_.size() != m_ptr_data_double.size() ) return false;
	vector<double>::iterator iter_double;
	vector<double *>::iterator iter_pdouble;
	for (iter_pdouble = m_ptr_data_double.begin(), iter_double = data_double_.begin(); iter_pdouble != m_ptr_data_double.end(); iter_pdouble++, iter_double++) {
		*(*iter_pdouble) = *iter_double;
	}
	return true;
}

bool SimulData::set_data_float(vector<float> &data_float_) 
{
	if ( data_float_.size() != m_ptr_data_float.size() ) return false;
	vector<float>::iterator iter_float;
	vector<float *>::iterator iter_pfloat;
	for (iter_pfloat = m_ptr_data_float.begin(), iter_float = data_float_.begin(); iter_pfloat != m_ptr_data_float.end(); iter_pfloat++, iter_float++) {
		*(*iter_pfloat) = *iter_float;
	}
	return true;
}

bool SimulData::set_data_int(vector<int> &data_int_)
{
	if ( data_int_.size() != m_ptr_data_int.size() ) return false;
	vector<int>::iterator iter_int;
	vector<int *>::iterator iter_pint;
	for (iter_pint = m_ptr_data_int.begin(), iter_int = data_int_.begin(); iter_pint != m_ptr_data_int.end(); iter_pint++, iter_int++) {
		*(*iter_pint) = *iter_int;
	}
	return true;
}

bool SimulData::set_data_bool(vector<bool> &data_bool_)
{
	if ( data_bool_.size() != m_ptr_data_bool.size() ) return false;
	vector<bool>::iterator iter_bool;
	vector<bool *>::iterator iter_pbool;
	for (iter_pbool = m_ptr_data_bool.begin(), iter_bool = data_bool_.begin(); iter_pbool != m_ptr_data_bool.end(); iter_pbool++, iter_bool++) {
		*(*iter_pbool) = *iter_bool;
	}
	return true;
}

vector<double> SimulData::get_data_double()
{
	vector<double> data_double(m_ptr_data_double.size());
	vector<double>::iterator iter_double;
	vector<double *>::iterator iter_pdouble;
	for (iter_pdouble = m_ptr_data_double.begin(), iter_double = data_double.begin(); iter_pdouble != m_ptr_data_double.end(); iter_pdouble++, iter_double++) {
		*iter_double = *(*iter_pdouble);
	}
	return data_double;
}

vector<float> SimulData::get_data_float()
{
	vector<float> data_float(m_ptr_data_float.size());
	vector<float>::iterator iter_float;
	vector<float *>::iterator iter_pfloat;
	for (iter_pfloat = m_ptr_data_float.begin(), iter_float = data_float.begin(); iter_pfloat != m_ptr_data_float.end(); iter_pfloat++, iter_float++) {
		*iter_float = *(*iter_pfloat);
	}
	return data_float;
}

vector<int> SimulData::get_data_int()
{
	vector<int> data_int(m_ptr_data_int.size());
	vector<int>::iterator iter_int;
	vector<int *>::iterator iter_pint;
	for (iter_pint = m_ptr_data_int.begin(), iter_int = data_int.begin(); iter_pint != m_ptr_data_int.end(); iter_pint++, iter_int++) {
		*iter_int = *(*iter_pint);
	}
	return data_int;
}

vector<bool> SimulData::get_data_bool()
{
	vector<bool> data_bool(m_ptr_data_bool.size());
	vector<bool>::iterator iter_bool;
	vector<bool *>::iterator iter_pbool;
	for (iter_pbool = m_ptr_data_bool.begin(), iter_bool = data_bool.begin(); iter_pbool != m_ptr_data_bool.end(); iter_pbool++, iter_bool++) {
		*iter_bool = *(*iter_pbool);
	}
	return data_bool;
}

void SimulData::set_filepath_for_writing(const char *fnamew_)
{
	m_filepath_w = string(fnamew_);
	m_filepath_log = m_filepath_w; m_filepath_log.append(".log");
}

void SimulData::set_filepath_for_reading(const char *fnamer_) 
{ 
	m_filepath_r = string(fnamer_); 
}

void SimulData::edit_num_data_set(const char *file_path_, int n_)
{
	ofstream fout;
	fout.open(file_path_, ios_base::binary | ios_base::in | ios_base::ate);
	fout.seekp(0, ios_base::beg);
	fout.write((char *)&n_, sizeof(int));
	fout.seekp(0, ios_base::end);
	fout.close();
}

void SimulData::get_time_sequence(vector<double> &ts_)
{
	ts_ = vector<double>(m_list_time.begin(), m_list_time.end());
}

void SimulData::get_time_sequence(vector<double> &ts_, const char *file_path_)
{
	ifstream fin;
	int n, nd, nf, ni, nb;

	fin.open(file_path_, ios::binary);

	if ( !fin.is_open() ) return;

	fin.read((char *)&n, sizeof(int));
	fin.read((char *)&nd, sizeof(int));
	fin.read((char *)&nf, sizeof(int));
	fin.read((char *)&ni, sizeof(int));
	fin.read((char *)&nb, sizeof(int));

	if ( nd != m_ptr_data_double.size() || nf != m_ptr_data_float.size() || ni != m_ptr_data_int.size() || nb != m_ptr_data_bool.size() ) return;

	double t;
	int skip = int(m_ptr_data_double.size()) * sizeof(double)
				+ int(m_ptr_data_float.size()) * sizeof(float)
				+ int(m_ptr_data_int.size()) * sizeof(int)
				+ int(m_ptr_data_bool.size()) * sizeof(bool);	// size of data without time

	ts_.clear();

	while (1) {
		if ( fin.eof() ) break;
		fin.read((char *)&t, sizeof(double));
		ts_.push_back(t);
		fin.seekg(skip, ios_base::cur);
	}

	fin.close();

	ts_.pop_back(); // why??
}

double SimulData::get_time_step()
{
	if ( m_list_time.size() <2 ) {
		return 0;
	}
	list<double>::iterator iter = m_list_time.begin();
	double t0 = *iter++;
	double t1 = *iter;
	return t1-t0;
}

//bool SimulData::_write_data(ofstream *fout_, vector<double *> &ptr_data_)
//{
//	vector<double *>::iterator iter_p;
//
//	for (iter_p = ptr_data_.begin(); iter_p != ptr_data_.end(); iter_p++) {
//		fout_->write((char *)(*iter_p), sizeof(double));
//	}
//
//	return true;
//}
//
//bool SimulData::_write_data(ofstream *fout_, vector<int *> &ptr_data_)
//{
//	vector<int *>::iterator iter_p;
//
//	for (iter_p = ptr_data_.begin(); iter_p != ptr_data_.end(); iter_p++) {
//		fout_->write((char *)(*iter_p), sizeof(int));
//	}
//
//	return true;
//}
//
//bool SimulData::_write_data(ofstream *fout_, vector<bool *> &ptr_data_)
//{
//	vector<bool *>::iterator iter_p;
//
//	for (iter_p = ptr_data_.begin(); iter_p != ptr_data_.end(); iter_p++) {
//		fout_->write((char *)(*iter_p), sizeof(bool));
//	}
//
//	return true;
//}

//bool SimulData::_read_data(ifstream *fin_, vector<double *> &ptr_data_)
//{
//	vector<double *>::iterator iter_p;
//
//	for (iter_p = ptr_data_.begin(); iter_p != ptr_data_.end(); iter_p++) {
//		fin_->read((char *)(*iter_p), sizeof(double));
//	}
//
//	return true;
//}
//
//bool SimulData::_read_data(ifstream *fin_, vector<int *> &ptr_data_)
//{
//	vector<int *>::iterator iter_p;
//
//	for (iter_p = ptr_data_.begin(); iter_p != ptr_data_.end(); iter_p++) {
//		fin_->read((char *)(*iter_p), sizeof(int));
//	}
//
//	return true;
//}
//
//bool SimulData::_read_data(ifstream *fin_, vector<bool *> &ptr_data_)
//{
//	vector<bool *>::iterator iter_p;
//
//	for (iter_p = ptr_data_.begin(); iter_p != ptr_data_.end(); iter_p++) {
//		fin_->read((char *)(*iter_p), sizeof(bool));
//	}
//
//	return true;
//}

bool SimulData::_write_data(ofstream *fout_, vector<double> &data_)
{
	if ( data_.size() == 0 ) return true;

	int i;
	vector<double>::iterator iter_p;

	for (i=0, iter_p = data_.begin(); iter_p != data_.end(); iter_p++, i++) {
		m_ptr_data_double_buffer[i] = *iter_p;
	}
	fout_->write((char *)m_ptr_data_double_buffer, m_ptr_data_double.size()*sizeof(double));

	return true;
}

bool SimulData::_write_data(ofstream *fout_, vector<float> &data_)
{
	if ( data_.size() == 0 ) return true;

	int i;
	vector<float>::iterator iter_p;

	for (i=0, iter_p = data_.begin(); iter_p != data_.end(); iter_p++, i++) {
		m_ptr_data_float_buffer[i] = *iter_p;
	}
	fout_->write((char *)m_ptr_data_float_buffer, m_ptr_data_float.size()*sizeof(float));

	return true;
}

bool SimulData::_write_data(ofstream *fout_, vector<int> &data_)
{
	if ( data_.size() == 0 ) return true;

	int i;
	vector<int>::iterator iter_p;

	for (i=0, iter_p = data_.begin(); iter_p != data_.end(); iter_p++, i++) {
		m_ptr_data_int_buffer[i] = *iter_p;
	}
	fout_->write((char *)m_ptr_data_int_buffer, m_ptr_data_int.size()*sizeof(int));

	return true;
}

bool SimulData::_write_data(ofstream *fout_, vector<bool> &data_)	
{
	if ( data_.size() == 0 ) return true;

	int i;
	vector<bool>::iterator iter_p;

	for (i=0, iter_p = data_.begin(); iter_p != data_.end(); iter_p++, i++) {
		m_ptr_data_bool_buffer[i] = *iter_p;
	}
	fout_->write((char *)m_ptr_data_bool_buffer, m_ptr_data_bool.size()*sizeof(bool));

	return true;
}

bool SimulData::_read_data(ifstream *fin_, vector<double> &data_)
{
	if ( data_.size() == 0 ) return true;

	int i;
	vector<double>::iterator iter_p;

	fin_->read((char *)m_ptr_data_double_buffer, m_ptr_data_double.size()*sizeof(double));
	for (i=0, iter_p = data_.begin(); iter_p != data_.end(); iter_p++, i++) {
		*iter_p = m_ptr_data_double_buffer[i];
	}

	return true;
}

bool SimulData::_read_data(ifstream *fin_, vector<float> &data_)
{
	if ( data_.size() == 0 ) return true;

	int i;
	vector<float>::iterator iter_p;

	fin_->read((char *)m_ptr_data_float_buffer, m_ptr_data_float.size()*sizeof(float));
	for (i=0, iter_p = data_.begin(); iter_p != data_.end(); iter_p++, i++) {
		*iter_p = m_ptr_data_float_buffer[i];
	}

	return true;
}

bool SimulData::_read_data(ifstream *fin_, vector<int> &data_)
{
	if ( data_.size() == 0 ) return true;

	int i;
	vector<int>::iterator iter_p;

	fin_->read((char *)m_ptr_data_int_buffer, m_ptr_data_int.size()*sizeof(int));
	for (i=0, iter_p = data_.begin(); iter_p != data_.end(); iter_p++, i++) {
		*iter_p = m_ptr_data_int_buffer[i];
	}

	return true;
}

bool SimulData::_read_data(ifstream *fin_, vector<bool> &data_)		
{
	if ( data_.size() == 0 ) return true;

	int i;
	vector<bool>::iterator iter_p;

	fin_->read((char *)m_ptr_data_bool_buffer, m_ptr_data_bool.size()*sizeof(bool));
	for (i=0, iter_p = data_.begin(); iter_p != data_.end(); iter_p++, i++) {
		*iter_p = m_ptr_data_bool_buffer[i];
	}

	return true;
}

bool SimulData::load_data_on_memory_from_file()
{
	if ( m_filepath_r.size() == 0 ) return false;

	ifstream fin;
	int n, nd, nf, ni, nb;

	fin.open(m_filepath_r.c_str(), ios::binary);

	if ( !fin.is_open() ) return false;

	fin.read((char *)&n, sizeof(int));
	fin.read((char *)&nd, sizeof(int));
	fin.read((char *)&nf, sizeof(int));
	fin.read((char *)&ni, sizeof(int));
	fin.read((char *)&nb, sizeof(int));

	if ( n < 0 || nd != m_ptr_data_double.size() || nf != m_ptr_data_float.size() || ni != m_ptr_data_int.size() || nb != m_ptr_data_bool.size() ) return false;

	m_list_time.resize(n);
	m_list_data_double.resize(n);
	m_list_data_float.resize(n);
	m_list_data_int.resize(n);
	m_list_data_bool.resize(n);

	list<double>::iterator iter_time = m_list_time.begin();
	list< vector<double> >::iterator iter_data_double = m_list_data_double.begin();
	list< vector<float> >::iterator iter_data_float = m_list_data_float.begin();
	list< vector<int> >::iterator iter_data_int = m_list_data_int.begin();
	list< vector<bool> >::iterator iter_data_bool = m_list_data_bool.begin();

	for (int i=0; i<n; i++) {
		(*iter_data_double).resize(nd);
		(*iter_data_float).resize(nf);
		(*iter_data_int).resize(ni);
		(*iter_data_bool).resize(nb);

		fin.read((char *)(&(*iter_time)), sizeof(double));
		_read_data(&fin, *iter_data_double);
		_read_data(&fin, *iter_data_float);
		_read_data(&fin, *iter_data_int);
		_read_data(&fin, *iter_data_bool);

		iter_time++;
		iter_data_double++;
		iter_data_float++;
		iter_data_int++;
		iter_data_bool++;
	}

	for (size_t i=0; i<m_vsdata_int.size(); i++) {
		if ( !m_vsdata_int[i].load_data_on_memory_from_file(&fin) ) return false;
	}
	for (size_t i=0; i<m_vsdata_float.size(); i++) {
		if ( !m_vsdata_float[i].load_data_on_memory_from_file(&fin) ) return false;
	}
	for (size_t i=0; i<m_vsdata_double.size(); i++) {
		if ( !m_vsdata_double[i].load_data_on_memory_from_file(&fin) ) return false;
	}
	for (size_t i=0; i<m_vsdata_Vec3.size(); i++) {
		if ( !m_vsdata_Vec3[i].load_data_on_memory_from_file(&fin) ) return false;
	}

	fin.close();
	
	return true;
}

bool SimulData::load_data_on_memory_from_file_app()
{
	if ( m_filepath_r.size() == 0 ) return false;

	ifstream fin;
	int n, nd, nf, ni, nb;

	fin.open(m_filepath_r.c_str(), ios::binary);

	if ( !fin.is_open() ) return false;

	fin.read((char *)&n, sizeof(int));
	fin.read((char *)&nd, sizeof(int));
	fin.read((char *)&nf, sizeof(int));
	fin.read((char *)&ni, sizeof(int));
	fin.read((char *)&nb, sizeof(int));

	if ( n < 0 || nd != m_ptr_data_double.size() || nf != m_ptr_data_float.size() || ni != m_ptr_data_int.size() || nb != m_ptr_data_bool.size() ) return false;

	std::list< double > app_time(n);
	std::list< std::vector<double> > app_data_double(n);
	std::list< std::vector<float> > app_data_float(n);
	std::list< std::vector<int> > app_data_int(n);
	std::list< std::vector<bool> > app_data_bool(n);

	list<double>::iterator iter_time = app_time.begin();
	list< vector<double> >::iterator iter_data_double = app_data_double.begin();
	list< vector<float> >::iterator iter_data_float = app_data_float.begin();
	list< vector<int> >::iterator iter_data_int = app_data_int.begin();
	list< vector<bool> >::iterator iter_data_bool = app_data_bool.begin();

	for (int i=0; i<n; i++) {
		(*iter_data_double).resize(nd);
		(*iter_data_float).resize(nf);
		(*iter_data_int).resize(ni);
		(*iter_data_bool).resize(nb);

		fin.read((char *)(&(*iter_time)), sizeof(double));
		_read_data(&fin, *iter_data_double);
		_read_data(&fin, *iter_data_float);
		_read_data(&fin, *iter_data_int);
		_read_data(&fin, *iter_data_bool);

		iter_time++;
		iter_data_double++;
		iter_data_float++;
		iter_data_int++;
		iter_data_bool++;
	}

	// append the data
	m_list_time.splice(m_list_time.end(), app_time);
	m_list_data_double.splice(m_list_data_double.end(), app_data_double);
	m_list_data_float.splice(m_list_data_float.end(), app_data_float);
	m_list_data_int.splice(m_list_data_int.end(), app_data_int);
	m_list_data_bool.splice(m_list_data_bool.end(), app_data_bool);
	
	for (size_t i=0; i<m_vsdata_int.size(); i++) {
		if ( !m_vsdata_int[i].load_data_on_memory_from_file_app(&fin) ) return false;
	}
	for (size_t i=0; i<m_vsdata_float.size(); i++) {
		if ( !m_vsdata_float[i].load_data_on_memory_from_file_app(&fin) ) return false;
	}
	for (size_t i=0; i<m_vsdata_double.size(); i++) {
		if ( !m_vsdata_double[i].load_data_on_memory_from_file_app(&fin) ) return false;
	}
	for (size_t i=0; i<m_vsdata_Vec3.size(); i++) {
		if ( !m_vsdata_Vec3[i].load_data_on_memory_from_file_app(&fin) ) return false;
	}

	fin.close();

	return true;
}

void SimulData::save_data_on_memory_into_file()
{
	if ( m_filepath_w.size() == 0 ) return;

	ofstream fout;
	int n, nd, nf, ni, nb;

	n = int(m_list_time.size());
	nd = int(m_ptr_data_double.size());
	nf = int(m_ptr_data_float.size());
	ni = int(m_ptr_data_int.size());
	nb = int(m_ptr_data_bool.size());

	fout.open(m_filepath_w.c_str(), ios::binary);

	fout.write((char *)&n, sizeof(int));
	fout.write((char *)&nd, sizeof(int));
	fout.write((char *)&nf, sizeof(int));
	fout.write((char *)&ni, sizeof(int));
	fout.write((char *)&nb, sizeof(int));

	list<double>::iterator iter_time = m_list_time.begin();
	list< vector<double> >::iterator iter_data_double = m_list_data_double.begin();
	list< vector<float> >::iterator iter_data_float = m_list_data_float.begin();
	list< vector<int> >::iterator iter_data_int = m_list_data_int.begin();
	list< vector<bool> >::iterator iter_data_bool = m_list_data_bool.begin();

	for (int i=0; i<n; i++) {
		fout.write((char *)(&(*iter_time)), sizeof(double));
		_write_data(&fout, *iter_data_double);
		_write_data(&fout, *iter_data_float);
		_write_data(&fout, *iter_data_int);
		_write_data(&fout, *iter_data_bool);

		iter_time++;
		iter_data_double++;
		iter_data_float++;
		iter_data_int++;
		iter_data_bool++;
	}

	for (size_t i=0; i<m_vsdata_int.size(); i++) {
		m_vsdata_int[i].save_data_on_memory_into_file(&fout);
	}
	for (size_t i=0; i<m_vsdata_float.size(); i++) {
		m_vsdata_float[i].save_data_on_memory_into_file(&fout);
	}
	for (size_t i=0; i<m_vsdata_double.size(); i++) {
		m_vsdata_double[i].save_data_on_memory_into_file(&fout);
	}
	for (size_t i=0; i<m_vsdata_Vec3.size(); i++) {
		m_vsdata_Vec3[i].save_data_on_memory_into_file(&fout);
	}

	fout.close();
}

void SimulData::write_data_into_memory(double t_)
{
	int nd, nf, ni, nb;

	nd = int(m_ptr_data_double.size());
	nf = int(m_ptr_data_float.size());
	ni = int(m_ptr_data_int.size());
	nb = int(m_ptr_data_bool.size());

	_data_double.resize(nd);
	_data_float.resize(nf);
	_data_int.resize(ni);
	_data_bool.resize(nb);

	for (int i=0; i<nd; i++) {
		_data_double[i] = *(m_ptr_data_double[i]);
	}
	for (int i=0; i<nf; i++) {
		_data_float[i] = *(m_ptr_data_float[i]);
	}
	for (int i=0; i<ni; i++) {
		_data_int[i] = *(m_ptr_data_int[i]);
	}
	for (int i=0; i<nb; i++) {
		_data_bool[i] = *(m_ptr_data_bool[i]);
	}

	m_list_time.push_back(t_);
	m_list_data_double.push_back(_data_double);
	m_list_data_float.push_back(_data_float);
	m_list_data_int.push_back(_data_int);
	m_list_data_bool.push_back(_data_bool);

	for (size_t i=0; i<m_vsdata_int.size(); i++) {
		m_vsdata_int[i].write_data_into_memory();
	}
	for (size_t i=0; i<m_vsdata_float.size(); i++) {
		m_vsdata_float[i].write_data_into_memory();
	}
	for (size_t i=0; i<m_vsdata_double.size(); i++) {
		m_vsdata_double[i].write_data_into_memory();
	}
	for (size_t i=0; i<m_vsdata_Vec3.size(); i++) {
		m_vsdata_Vec3[i].write_data_into_memory();
	}
}

bool SimulData::read_data_from_memory(double &t_, int idx_)
{
	int nd, nf, ni, nb;

	nd = int(m_ptr_data_double.size());
	nf = int(m_ptr_data_float.size());
	ni = int(m_ptr_data_int.size());
	nb = int(m_ptr_data_bool.size());

	list<double>::iterator iter_time = m_list_time.begin();
	list< vector<double> >::iterator iter_data_double = m_list_data_double.begin();
	list< vector<float> >::iterator iter_data_float = m_list_data_float.begin();
	list< vector<int> >::iterator iter_data_int = m_list_data_int.begin();
	list< vector<bool> >::iterator iter_data_bool = m_list_data_bool.begin();

	for (int i=0; i<idx_; i++) {
		iter_time++;
		iter_data_double++;
		iter_data_float++;
		iter_data_int++;
		iter_data_bool++;
	}

	t_ = *iter_time;
	for (int i=0; i<nd; i++) {
		*m_ptr_data_double[i] = (*iter_data_double)[i];
	}
	for (int i=0; i<nf; i++) {
		*m_ptr_data_float[i] = (*iter_data_float)[i];
	}
	for (int i=0; i<ni; i++) {
		*m_ptr_data_int[i] = (*iter_data_int)[i];
	}
	for (int i=0; i<nb; i++) {
		*m_ptr_data_bool[i] = (*iter_data_bool)[i];
	}

	for (size_t i=0; i<m_vsdata_int.size(); i++) {
		m_vsdata_int[i].set_data_all(0);
		m_vsdata_int[i].read_data_from_memory(idx_);
	}
	for (size_t i=0; i<m_vsdata_float.size(); i++) {
		m_vsdata_float[i].set_data_all(0);
		m_vsdata_float[i].read_data_from_memory(idx_);
	}
	for (size_t i=0; i<m_vsdata_double.size(); i++) {
		m_vsdata_double[i].set_data_all(0);
		m_vsdata_double[i].read_data_from_memory(idx_);
	}
	for (size_t i=0; i<m_vsdata_Vec3.size(); i++) {
		m_vsdata_Vec3[i].set_data_all(Vec3(0,0,0));
		m_vsdata_Vec3[i].read_data_from_memory(idx_);
	}

	return true;
}

void SimulData::clear_data_on_memory()
{
	m_list_time.clear();
	m_list_data_double.clear();
	m_list_data_float.clear();
	m_list_data_int.clear();
	m_list_data_bool.clear();

	for (size_t i=0; i<m_vsdata_int.size(); i++) {
		m_vsdata_int[i].clear_memory();
	}
	for (size_t i=0; i<m_vsdata_float.size(); i++) {
		m_vsdata_float[i].clear_memory();
	}
	for (size_t i=0; i<m_vsdata_double.size(); i++) {
		m_vsdata_double[i].clear_memory();
	}
	for (size_t i=0; i<m_vsdata_Vec3.size(); i++) {
		m_vsdata_Vec3[i].clear_memory();
	}
}

void SimulData::truncate_data_on_memory(int idx_)
{
	if ( idx_ >= (int)m_list_time.size() ) return;

	list<double>::iterator iter_time = m_list_time.begin();
	list< vector<double> >::iterator iter_data_double = m_list_data_double.begin();
	list< vector<float> >::iterator iter_data_float = m_list_data_float.begin();
	list< vector<int> >::iterator iter_data_int = m_list_data_int.begin();
	list< vector<bool> >::iterator iter_data_bool = m_list_data_bool.begin();

	std::advance(iter_time, idx_);
	std::advance(iter_data_double, idx_);
	std::advance(iter_data_float, idx_);
	std::advance(iter_data_int, idx_);
	std::advance(iter_data_bool, idx_);

	//for (int i=0; i<idx_; i++) {
	//	iter_time++;
	//	iter_data_double++;
	//	iter_data_float++;
	//	iter_data_int++;
	//	iter_data_bool++;
	//}

	m_list_time.erase(iter_time, m_list_time.end());
	m_list_data_double.erase(iter_data_double, m_list_data_double.end());
	m_list_data_float.erase(iter_data_float, m_list_data_float.end());
	m_list_data_int.erase(iter_data_int, m_list_data_int.end());
	m_list_data_bool.erase(iter_data_bool, m_list_data_bool.end());

	for (size_t i=0; i<m_vsdata_int.size(); i++) {
		m_vsdata_int[i].truncate_data_on_memory(idx_);
	}
	for (size_t i=0; i<m_vsdata_float.size(); i++) {
		m_vsdata_float[i].truncate_data_on_memory(idx_);
	}
	for (size_t i=0; i<m_vsdata_double.size(); i++) {
		m_vsdata_double[i].truncate_data_on_memory(idx_);
	}
	for (size_t i=0; i<m_vsdata_Vec3.size(); i++) {
		m_vsdata_Vec3[i].truncate_data_on_memory(idx_);
	}

}

double SimulData::get_time_from_data_on_memory(int idx_)
{
	list<double>::iterator iter_time = m_list_time.begin();
	for (int i=0; i<idx_; i++) {
		iter_time++;
	}
	return *iter_time;
}

