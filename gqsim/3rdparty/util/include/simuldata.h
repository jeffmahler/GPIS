//================================================================================
//         DATA SAVER FOR SIMULATION
// 
//                                                               junggon@gmail.com
//================================================================================

#ifndef _SIMUL_DATA_
#define _SIMUL_DATA_

#include <fstream>
#include <vector>
#include <list>
#include <string>
#include "datasaver.h"
#include "liegroup.h"

//===================================================================
//			SimulData
//===================================================================
class SimulData
{
public:
	// pointers of data to be saved
	std::vector<double *> m_ptr_data_double;
	std::vector<float *> m_ptr_data_float;
	std::vector<int *> m_ptr_data_int;
	std::vector<bool *> m_ptr_data_bool;

	// file path for writing/reading data 
	std::string m_filepath_w, m_filepath_r, m_filepath_log;	

	// reading/writing data from/into memory 
	std::list< double > m_list_time;
	std::list< std::vector<double> > m_list_data_double;
	std::list< std::vector<float> > m_list_data_float;
	std::list< std::vector<int> > m_list_data_int;
	std::list< std::vector<bool> > m_list_data_bool;

	// buffer on consecutive memory for reading/writing data from/into file
	double *m_ptr_data_double_buffer;
	float *m_ptr_data_float_buffer;
	int *m_ptr_data_int_buffer;
	bool *m_ptr_data_bool_buffer;

	// varying data
	std::vector< DataSaverVaryingSize<int> > m_vsdata_int;
	std::vector< DataSaverVaryingSize<float> > m_vsdata_float;
	std::vector< DataSaverVaryingSize<double> > m_vsdata_double;
	std::vector< DataSaverVaryingSize< Vec3 > > m_vsdata_Vec3;

public:
	SimulData() { m_ptr_data_double_buffer = NULL; m_ptr_data_float_buffer = NULL; m_ptr_data_int_buffer = NULL; m_ptr_data_bool_buffer = NULL; }
	~SimulData() { clear_data_address_all(); }

public:
	int size() { return m_list_time.size(); }

	// ------------ data list ------------------------------------------------------

	void clear_data_address_all() 
	{
		if ( m_ptr_data_double_buffer != NULL ) delete [] m_ptr_data_double_buffer; 
		if ( m_ptr_data_float_buffer != NULL ) delete [] m_ptr_data_float_buffer;
		if ( m_ptr_data_int_buffer != NULL ) delete [] m_ptr_data_int_buffer; 
		if ( m_ptr_data_bool_buffer != NULL ) delete [] m_ptr_data_bool_buffer; 
	}

	void set_data_address_double(std::vector<double *> ptr_data_double_) { m_ptr_data_double = ptr_data_double_; m_ptr_data_double_buffer = new double [ptr_data_double_.size()]; }
	void set_data_address_float(std::vector<float *> ptr_data_float_) { m_ptr_data_float = ptr_data_float_; m_ptr_data_float_buffer = new float [ptr_data_float_.size()]; }
	void set_data_address_int(std::vector<int *> ptr_data_int_) { m_ptr_data_int = ptr_data_int_; m_ptr_data_int_buffer = new int [ptr_data_int_.size()]; }
	void set_data_address_bool(std::vector<bool *> ptr_data_bool_) { m_ptr_data_bool = ptr_data_bool_; m_ptr_data_bool_buffer = new bool [ptr_data_bool_.size()]; }

	void add_varying_data_int(std::vector<int> *ptr_data, std::vector<int> *ptr_indices)
	{
		m_vsdata_int.push_back(DataSaverVaryingSize<int>(ptr_data, ptr_indices));
	}

	void add_varying_data_float(std::vector<float> *ptr_data, std::vector<int> *ptr_indices)
	{
		m_vsdata_float.push_back(DataSaverVaryingSize<float>(ptr_data, ptr_indices));
	}

	void add_varying_data_double(std::vector<double> *ptr_data, std::vector<int> *ptr_indices)
	{
		m_vsdata_double.push_back(DataSaverVaryingSize<double>(ptr_data, ptr_indices));
	}

	void add_varying_data_Vec3(std::vector<Vec3> *ptr_data, std::vector<int> *ptr_indices)
	{
		m_vsdata_Vec3.push_back(DataSaverVaryingSize<Vec3>(ptr_data, ptr_indices));
	}

	bool set_data_double(std::vector<double> &data_double_);
	bool set_data_float(std::vector<float> &data_float_);
	bool set_data_int(std::vector<int> &data_int_);
	bool set_data_bool(std::vector<bool> &data_bool_);

	std::vector<double> get_data_double();
	std::vector<float> get_data_float();
	std::vector<int> get_data_int();
	std::vector<bool> get_data_bool();

	// ------------ file path for writing/reading data -----------------------------

	// ------data file format--------
	// n											// number of data set
	// d f i b										// number of data_double, data_float, data_int, data_bool
	// t[0]  d[0][0] d[0][1] ... d[0][m-1]			// time, data_double, data_int, data_bool
	// t[1]  d[1][0] d[1][1] ... d[1][m-1]
	//       .....................
	// t[n-1] d[n-1][0] d[n-1][1] ... d[n-1][m-1]
	//
	// --------------------------------

	void set_filepath_for_writing(const char *fnamew_); 
	void set_filepath_for_reading(const char *fnamer_);

	const char* get_filepath_for_writing() { return m_filepath_w.c_str(); }
	const char* get_filepath_for_reading() { return m_filepath_r.c_str(); }


	// ------------ reading/writing data from/into memory --------------------------

	// writing data into memory
	void write_data_into_memory(double t_);				// push back data

	// reading data from memory
	bool read_data_from_memory(double &t_, int idx_);	// read the idx-th data

	// etc
	void clear_data_on_memory();						// clear memory for data
	void truncate_data_on_memory(int idx_);				// truncate data (erase data[idx], data[idx+1], ...)
	double get_time_from_data_on_memory(int idx_);		// get time for data[idx]

	// load from file
	bool load_data_on_memory_from_file();
	bool load_data_on_memory_from_file_app();

	// save to file
	void save_data_on_memory_into_file();
	bool save_data_on_memory_into_file_rng(const char *file_path_, int idx1_, int idx2_, bool b_zero_start_time_);
	void save_data_on_memory_into_file_app(const char *file_path_, bool b_skip_first);

	// get time sequence
	void get_time_sequence(std::vector<double> &ts_);							// extract time sequence from the current data set
	void get_time_sequence(std::vector<double> &ts_, const char *file_path_);	// extract time sequence from the data file "file_path_"

	// get time step
	double get_time_step(); // return m_list_time[1]-m_list_time[0]


	// ------------ sub-functions for writing and reading --------------------------
	
	bool _write_data(std::ofstream *fout_, std::vector<double> &data_);
	bool _write_data(std::ofstream *fout_, std::vector<float> &data_);
	bool _write_data(std::ofstream *fout_, std::vector<int> &data_);
	bool _write_data(std::ofstream *fout_, std::vector<bool> &data_);
	bool _read_data(std::ifstream *fin_, std::vector<double> &data_);
	bool _read_data(std::ifstream *fin_, std::vector<float> &data_);
	bool _read_data(std::ifstream *fin_, std::vector<int> &data_);
	bool _read_data(std::ifstream *fin_, std::vector<bool> &data_);
	void edit_num_data_set(const char *file_path_, int n_);						// change the number of data set of the data file "file_path_"
};

#endif

