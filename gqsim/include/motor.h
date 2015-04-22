#ifndef _MOTOR_
#define _MOTOR_

#include <vector>
#include <string>
#include "utilfunc.h" // for moving average filter

class Motor
{
public:
	// properties
	std::string _name;
	double _stalltorque;		// stall (or maximum) torque
	double _maxspeed;			// maximum speed
	double _t_stop_duration;	// time for deciding overload (by seeing whether torque exceeds the stall torque for more than _t_stop_duration or not)
	double _t_halt_duration;	// time for deciding halt (by seeing whether speed is below a minimum threshold for more than _t_halt_duration or not)
	bool _is_enabled_autolock;	// set true to lock the motor if torque exceeds _stalltorque * _emergency_stop_ratio
	double _emergency_stop_ratio; // emergency stop torque ratio to the stall torque

	// status
	bool _is_closed;			// true if motor closed (to send a message if the finger closing has been done.)
	bool _is_locked;			// true if motor locked
	bool _is_under_breakaway_condition; // true if motor is under breakaway condition

	// joints driven by the motor
	std::vector<GCoordinate*> _pjointcoords;
	std::vector<double> _ratios;

	// breakaway system
	bool _is_enabled_breakaway;
	double _breakawaytorque;
	std::vector<double> _ratios_switched;

	// closing direction
	int _closingdirection; // colsing direction (0, 1 or -1)

	// motor speed for closing/opening (actual closing speed changes depending on the motor torque)
	double _speed;

	// target motor position for closing (rad)
	double _target_position;

	// a moving average filter for computing motor torque
	MovingAverageFilter _torque_filter;

	// internal variables
	bool _is_stalltorque_detected;			// true if torque exceeds the stall torque
	bool _is_breakawaytorque_detected;		// true if torque exceeds the breakaway torque
	bool _is_verysmallspeed_detected;		// true if closing speed is very small
	double _t_stalltorque_detected;			// saves the simulation time at which torque exceeds the stall torque
	double _t_breakawaytorque_detected;		// saves the simulation time at which torque excees the breakaway torque
	double _t_verysmallspeed_detected;		// saves the simulation time at which closing speed is very small

	// for debug
	bool _b_show_message;
	std::vector<double> _data_debugging;

	// constructor/destructor
	Motor() : _stalltorque(0), _maxspeed(0), _is_enabled_autolock(false), _t_stop_duration(0.030), _t_halt_duration(0.5), _emergency_stop_ratio(2.0), _is_closed(false), _is_locked(false), _is_stalltorque_detected(false), _t_stalltorque_detected(0), _is_enabled_breakaway(false), _is_under_breakaway_condition(false), _breakawaytorque(0), _closingdirection(0), _speed(0), _target_position(1E10), _torque_filter(30), _b_show_message(false), _data_debugging(10,0.0) {}
	~Motor() {}

	// set motor properties
	bool setMotor(std::string name, double maxspeed, double stalltorque, std::vector<GCoordinate*> pjointcoords, std::vector<double> ratios, bool autolock) 
	{
		if ( pjointcoords.size() != ratios.size() ) return false;
		_name = name; _maxspeed = maxspeed; _stalltorque = stalltorque; _pjointcoords = pjointcoords; _ratios = ratios; _is_enabled_autolock = autolock;
		return true;
	}
	
	// set breakaway
	bool setBreakAway(double breakawaytorque, std::vector<double> ratios2) 
	{
		if ( ratios2.size() != _pjointcoords.size() ) return false;
		_is_enabled_breakaway = true;
		_is_under_breakaway_condition = false;
		_breakawaytorque = breakawaytorque;
		_ratios_switched = ratios2;
		return true;
	}

	// for finger closing
	void setClosingDirection(int d) { _closingdirection = d; }
	void setSpeed(double s) { _speed = s; }
	void setTargetPosition(double position) { _target_position = position; } // position = _pjointcoords[0]->q)_target / _ratios[0]
	void setTorqueFilterBufferSize(int n) { _torque_filter.setBufferSize(n); }

	// is the motor locked?
	bool isLocked() { return _is_locked; }

	// is closing done?
	bool isClosingDone() { return _is_closed || _is_locked; }

	// init
	void init() 
	{ 
		_is_closed = _is_locked = _is_stalltorque_detected = _is_under_breakaway_condition = false;
		_is_stalltorque_detected = _is_breakawaytorque_detected = _is_verysmallspeed_detected = false; 
		_t_stalltorque_detected = _t_breakawaytorque_detected = _t_verysmallspeed_detected = 0;
		_torque_filter.clearBuffer(); 
		for (size_t i=0; i<_pjointcoords.size(); i++) { 
			_pjointcoords[i]->dq = 0;
			_pjointcoords[i]->ddq = 0;
			_pjointcoords[i]->tau = 0;
		} 
	}

	// open
	void open(double h)
	{
		for (size_t i=0; i<_pjointcoords.size(); i++) {
			_pjointcoords[i]->dq = -1.0 * double(_closingdirection) * _speed * _ratios[i];
			_pjointcoords[i]->q += h * _pjointcoords[i]->dq;
		}

		// consider joint limits
		for (size_t i=0; i<_pjointcoords.size(); i++) {
			if ( _pjointcoords[i]->q >= _pjointcoords[i]->qUL ) {
				_pjointcoords[i]->dq = 0;
				_pjointcoords[i]->q = _pjointcoords[i]->qUL;
			}
			if ( _pjointcoords[i]->q <= _pjointcoords[i]->qLL ) {
				_pjointcoords[i]->dq = 0;
				_pjointcoords[i]->q = _pjointcoords[i]->qLL;
			}
		}
	}

	// close (h = step size, cur_time = current simulation time)
	void close(double h, double cur_time) 
	{
		if ( _pjointcoords.size() == 0 ) return;
		if ( _closingdirection == 0 ) return;

		// compute motor torque by filtering joint torque
		double torque = fabs(_torque_filter.getValue(_ratios[0] * _pjointcoords[0]->tau));

		// monitor if torque exceeds the stall torque
		if ( !_is_stalltorque_detected && torque >= _stalltorque ) { 
			_is_stalltorque_detected = true;
			_t_stalltorque_detected = cur_time;
		} else if ( _is_stalltorque_detected && torque < _stalltorque ) {
			_is_stalltorque_detected = false;
		}

		// monitor if torque exceeds the breakaway torque
		if ( !_is_breakawaytorque_detected && torque >= _breakawaytorque ) { 
			_is_breakawaytorque_detected = true;
			_t_breakawaytorque_detected = cur_time;
		} else if ( _is_breakawaytorque_detected && torque < _breakawaytorque ) {
			_is_breakawaytorque_detected = false;
		}

		// if torque exceeds the stall torque for more than _t_stop_duration
		if ( _is_stalltorque_detected && cur_time-_t_stalltorque_detected > _t_stop_duration ) {

			// if enabled, lock the motor
			if ( _is_enabled_autolock && !_is_locked ) {
				_is_locked = true; // once locked, it will not be closing any more (use init() to start a new simulation)
				if ( _b_show_message ) {
					std::cout << "motor " << _name << ":: stopped at " << cur_time << " sec " << " :: stalled " << std::endl;
				}
			} 
			// or, just send a message that the motor has been closed
			else if (!_is_enabled_autolock && !_is_closed ) {
				_is_closed = true; // send a message that the motor has been closed (but, the motor is still under control)
				if ( _b_show_message ) {
					std::cout << "motor " << _name << ":: closing done at " << cur_time << " sec " << std::endl;
				}
			}
		}

		// if torque exceeds the breakaway torque for more than _t_stop_duration
		if ( _is_breakawaytorque_detected && cur_time-_t_breakawaytorque_detected > _t_stop_duration ) {
			// if enabled, breakaway
			if ( _is_enabled_breakaway && !_is_under_breakaway_condition ) {
				_is_under_breakaway_condition = true; // once breakaway has been activated, it will not be deactivated during the closing (use init() to start a new simulation)
				if ( _b_show_message ) {
					std::cout << "motor " << _name << ":: breakaway activated at " << cur_time << " sec " << std::endl;
				}
			}
		}

		// emergency stop (lock the motor if too large torque detected)
		if ( _is_enabled_autolock && !_is_locked && _is_stalltorque_detected && torque >= _emergency_stop_ratio * _stalltorque ) {
			_is_locked = true; // once stopped, it will not be closing any more (use init() to start a new simulation)
			if ( _b_show_message ) {
				std::cout << "motor " << _name << ":: locked at " << cur_time << " sec " << " :: emergency stop (too large torque detected!)" << std::endl;
			}
		}

		// if locked, set joint velocity to zero and return!
		if ( _is_locked ) {
			for (size_t i=0; i<_pjointcoords.size(); i++) {
				_pjointcoords[i]->dq = 0;
			}
			return;
		}

		// control the closing speed (using inverse linear relationship between the speed and torque)
		double current_speed = _speed, available_speed = 0;
		if ( torque < _stalltorque ) {
			available_speed = _maxspeed * fabs(1. - torque / _stalltorque);
		}
		if ( _is_enabled_autolock && available_speed < 0.01 * _speed ) {
			available_speed = 0.01 * _speed;
		}
		if ( current_speed > available_speed ) {
			current_speed = available_speed;
		}

		// drive the joints
		if ( _is_under_breakaway_condition ) {
			for (size_t i=0; i<_pjointcoords.size(); i++) {
				_pjointcoords[i]->dq = double(_closingdirection) * current_speed * _ratios_switched[i];
				_pjointcoords[i]->q += h * _pjointcoords[i]->dq;
			}
		} else {
			for (size_t i=0; i<_pjointcoords.size(); i++) {
				_pjointcoords[i]->dq = double(_closingdirection) * current_speed * _ratios[i];
				_pjointcoords[i]->q += h * _pjointcoords[i]->dq;
			}
		}

		// if reached target position
		if ( _pjointcoords.size() > 0 && _ratios.size() > 0 && fabs(_ratios[0]) > 1E-8 ) {
			if ( ( _closingdirection > 0 && _pjointcoords[0]->q / _ratios[0] > _target_position ) || ( _closingdirection < 0 && _ratios[0] * _pjointcoords[0]->q < _target_position ) ) {
				_is_locked = true;
				if ( _b_show_message ) {
					std::cout << "motor " << _name << ":: locked at " << cur_time << " sec " << " :: target position" << std::endl;
				}
			}
		}

		// consider joint limits
		for (size_t i=0; i<_pjointcoords.size(); i++) {
			if ( _pjointcoords[i]->q >= _pjointcoords[i]->qUL ) {
				_pjointcoords[i]->dq = 0;
				_pjointcoords[i]->q = _pjointcoords[i]->qUL;
				_is_locked = true; // if any of the joints has reached the limit, lock the motor.
				if ( _b_show_message ) {
					std::cout << "motor " << _name << ":: locked at " << cur_time << " sec " << " :: joint limit" << std::endl;
				}
			}
			if ( _pjointcoords[i]->q <= _pjointcoords[i]->qLL ) {
				_pjointcoords[i]->dq = 0;
				_pjointcoords[i]->q = _pjointcoords[i]->qLL;
				_is_locked = true; // if any of the joints has reached the limit, lock the motor.
				if ( _b_show_message ) {
					std::cout << "motor " << _name << ":: locked at " << cur_time << " sec " << " :: joint limit" << std::endl;
				}
			}
		}

		// monitor if closing speed is below the limit
		if ( !_is_verysmallspeed_detected && current_speed <= 0.01 * _speed ) {
			_is_verysmallspeed_detected = true;
			_t_verysmallspeed_detected = cur_time;
		} else if ( _is_verysmallspeed_detected && current_speed > 0.01 * _speed ) {
			_is_verysmallspeed_detected = false;
		}

		// if closing speed has been very small for more than 1 sec
		if ( _is_verysmallspeed_detected && cur_time-_t_verysmallspeed_detected > _t_halt_duration ) {
			if ( !_is_closed ) {
				_is_closed = true; // send a message that the motor has been closed (but, the motor is still under control)
				if ( _b_show_message ) {
					std::cout << "motor " << _name << ":: closing done at " << cur_time << " sec " << std::endl;
				}
			}
		}

		// for debugging
		_data_debugging[0] = cur_time;
		_data_debugging[1] = torque;
		_data_debugging[2] = current_speed;
		if ( _pjointcoords.size() > 0 ) {
			_data_debugging[3] = _pjointcoords[0]->q;
		}

	}

	friend std::ostream &operator << (std::ostream &os, const Motor &m)
	{
		os << "Motor:: name = " << m._name << std::endl;
		os << "  max speed = " << m._maxspeed << std::endl;
		os << "  stall torque = " << m._stalltorque << std::endl;
		os << "  is stopped? " << m._is_locked << std::endl;
		os << "  number of joint coordinates connected to this motor = " << m._pjointcoords.size() << std::endl;
		os << "  reduction ratios = "; for (size_t i=0; i<m._ratios.size(); i++) { os << m._ratios[i] << ", "; } os << std::endl;
		os << "  breakaway enabled? " << m._is_enabled_breakaway << std::endl;
		os << "  under breakaway? " << m._is_under_breakaway_condition << std::endl;
		os << "  breakaway torque = " << m._breakawaytorque << std::endl;
		os << "  reduction ratios in breakaway = "; for (size_t i=0; i<m._ratios_switched.size(); i++) { os << m._ratios_switched[i] << ", "; } os << std::endl;
		os << "  closing direction = " << m._closingdirection << std::endl;
		os << "  closing speed = " << m._speed << std::endl;
		return os;
	}

};

#endif

