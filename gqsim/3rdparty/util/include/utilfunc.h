//================================================================================
//         UTILITY TOOLS
// 
//                                                               junggon@gmail.com
//================================================================================

#ifndef _UTILITY_FUNCTIONS_
#define _UTILITY_FUNCTIONS_

#include <list>
#include <fstream>

// randn(m,s) generates a random number from normal distribution (m = mean, s = standard deviation)
double randn(double m, double s);

// Moving Average Filter
class MovingAverageFilter
{
private:
	std::list<double> _buffer;
	double _sum;

public:
	MovingAverageFilter() : _buffer(10, 0.0), _sum(0) {}
	MovingAverageFilter(int n) : _buffer(n, 0.0), _sum(0) {}
	~MovingAverageFilter() {}

	void setBufferSize(int n) { _buffer.resize(n); clearBuffer(); }

	void clearBuffer() { _buffer.assign(_buffer.size(), 0.0); _sum = 0; }

	double getValue() { return _sum / double(_buffer.size()); }

	double getValue(double in) { pushValue(in); return getValue(); }

	void pushValue(double in)
	{
		_sum -= _buffer.front();
		_buffer.pop_front();
		_buffer.push_back(in);
		_sum += in;
	}
};

bool fexists(const char *filepath);

#endif

