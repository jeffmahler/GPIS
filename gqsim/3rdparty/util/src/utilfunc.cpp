//================================================================================
//         UTILITY TOOLS
// 
//                                                               junggon@gmail.com
//================================================================================

#include <math.h>
#include "utilfunc.h"
#include "rmatrix3j.h"

// randn(m,s) generates a random number from normal distribution (m = mean, s = standard deviation)
double randn(double m, double s) 
{
	double x1, x2, w, y1;
	static double _y2;
	static bool _b_last = false;

	if (_b_last) {
		y1 = _y2;
		_b_last = false;
	} else {
		do {
			x1 = 2.0 * prand() - 1.0;
			x2 = 2.0 * prand() - 1.0;
			w = x1 * x1 + x2 * x2;
		} while ( w >= 1.0 );

		w = sqrt( (-2.0 * log( w ) ) / w );
		y1 = x1 * w;
		_y2 = x2 * w;
		_b_last = true;
	}

	return( m + y1 * s );
}

bool fexists(const char *filepath)
{
	std::ifstream fin(filepath);
	return fin;
}

