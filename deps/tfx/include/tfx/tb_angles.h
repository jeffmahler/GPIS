/*
 * Copyright 2015 Ben Kehoe
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#ifndef TFX_TB_ANGLES_H_
#define TFX_TB_ANGLES_H_

#include <math.h>
#include <string>
#include <stdio.h>

#include <LinearMath/btTransform.h>
#include <LinearMath/btMatrix3x3.h>
#include <LinearMath/btQuaternion.h>

#include <geometry_msgs/Quaternion.h>

#ifndef TB_ANGLES_CLOSE_ENOUGH
#define TB_ANGLES_CLOSE_ENOUGH 0.0001
#endif

namespace tfx {

#ifndef TFX_FIX_ANGLE
#define TFX_FIX_ANGLE
/**
 * Utility function to fix an angle to be in a certain 2 pi range.
 * By default, the range is -pi to pi, i.e., a center of 0.
 */
inline float fix_angle(float angle,float center = 0) {
	float test_angle = angle;
	int cnt = 1;
	while ((test_angle-center) > M_PI) {
		test_angle = angle - cnt * 2*M_PI;
		cnt++;
	}
	angle = test_angle;
	cnt = 1;
	while ((test_angle-center) < -M_PI) {
		test_angle = angle + cnt * 2*M_PI;
		cnt++;
	}
	return test_angle;
}
#endif

/**
 * Tait-Bryan angles, aka aircraft angles (intrinsic yaw pitch roll)
 *
 * This class is intended to simplify human interaction with rotations.
 * Note that all values in this class are, by default, given and displayed
 * in degrees, and that the class is designed with mutability in mind.
 *
 * Quaternions and rotation matrices are computationally efficient, but cannot
 * be easily visualized by inspecting their values. Even axis-angle
 * representations can be difficult to visualize immediately.
 * Additionally, given a known physical rotation, it is difficult to generate
 * the quaternion or rotation matrix corresponding to that rotation.
 *
 * However, Tait-Bryan angles, also known as aircraft angles, are very easy to
 * visualize. It is an intrinsic (moving-axis) yaw-pitch-roll (zyx) rotation.
 * They are very similar to Euler angles, and in fact are often referred to as
 * Euler angles. However, the term Tait-Bryan angles is used here to clearly
 * distinguish them from other forms of Euler angles.
 *
 * Given yaw, pitch, and roll angles, the rotation is sequentially a rotation
 * about the z axis (yaw), then a rotation about the rotated y axis (pitch),
 * and finally a rotation about the rotated x axis (roll). The first two
 * rotations correspond to the angular axes in spherical coordinates, and the
 * final rotation is about the third (radial) axis.
 *
 * The class can be created with angles (in degrees by default) or with a
 * Bullet quaternion, matrix, or transform, or a ROS message.
 *
 * Examples of creating with angles:
 * tb_angles(90,45,-90)
 * tb_angles(pi/2,pi/4,-pi/2,tb_angles::RAD)
 *
 * The class is not immutable, but modifying the angles after creation is not
 * recommended. If a degree field is changed, updateFromDegrees() should be
 * called to set the radians fields from the degree fields, and
 * updateFromRadians() should be called in the reverse case.
 *
 * tb_angles can be created with the following rotation formats:
 * Quaternion as btQuaternion or geometry_msgs/Quaternion
 * Rotation matrix as btMatrix3x3
 * Transformation matrix (translation ignored) as btTransform
 * Any matrix that can be accessed as M[i][j]
 *
 * Converting tb_angles to rotation formats:
 *
 * Bullet formats:
 * toQuaternion()
 * toMatrix()
 * toTransform()
 *
 * ROS geometry_msgs/Quaternion:
 * toMsg()
 *
 * Printing the angles:
 * The toString() method has various options for formatting the string output.
 * The << operator is overloaded for tb_angles, calling the toString() method.
 *
 * By default, the values are shown in degrees, without units.
 * The options tb_angles::DEG and tb_angles::RAD can be used to display the
 * values in degrees and radians, with units.
 * tb.tostring():              values in degrees without units
 * tb.tostring(tb_angles::DEG) values in degrees    with units
 * tb.tostring(tb_angles::RAD) values in radians    with units
 * tb.tostring(tb_angles::DEG & tb_angles::RAD) will print values in both degrees
 *     and radians
 *
 * The option tb_angles::FIXED_WIDTH will cause the output to use fixed-width
 * fields for the values, which can be useful if many values are being
 * printed in succession.
 *
 * The option tb_angles::SHORT_NAMES causes the field names to be abbreviated
 * to one letter, e.g., 'y' instead of 'yaw'
 */
class tb_angles {
public:
	/**
	 * The yaw angle in degrees. If modified, call updateFromDegrees()
	 */
	float yaw_deg;
	/**
	 * The yaw angle in radians. If modified, call updateFromRadians()
	 */
	float yaw_rad;
	/**
	 * The pitch angle in degrees. If modified, call updateFromDegrees()
	 */
	float pitch_deg;
	/**
	 * The pitch angle in radians. If modified, call updateFromRadians()
	 */
	float pitch_rad;
	/**
	 * The roll angle in degrees. If modified, call updateFromDegrees()
	 */
	float roll_deg;
	/**
	 * The roll angle in radians. If modified, call updateFromRadians()
	 */
	float roll_rad;

	enum angle_type { DEG = 1, RAD = 2 };

	/**
	 * Create tb_angles using a yaw, pitch and roll angles, in degrees by
	 * default. Set angle_type to tb_angles::RAD if giving angles in radians.
	 */
	tb_angles(float yaw, float pitch, float roll,int angle_type=DEG) {
		if (angle_type & RAD) {
			this->yaw_rad = yaw;
			this->yaw_deg = yaw * 180. / M_PI;
			this->pitch_rad = pitch;
			this->pitch_deg = pitch * 180. / M_PI;
			this->roll_rad = roll;
			this->roll_deg = roll * 180. / M_PI;
		} else {
			this->yaw_rad = yaw * M_PI / 180.;
			this->yaw_deg = yaw;
			this->pitch_rad = pitch * M_PI / 180.;
			this->pitch_deg = pitch;
			this->roll_rad = roll * M_PI / 180.;
			this->roll_deg = roll;
		}
	}
	/**
	 * Create tb_angles from a btQuaternion.
	 */
	tb_angles(const btQuaternion& q) { init(btMatrix3x3(q)); }
	/**
	 * Create tb_angles from a btTransform.
	 */
	tb_angles(const btTransform& T) { init(T.getBasis()); }
	/**
	 * Create tb_angles from a btMatrix3x3.
	 */
	tb_angles(const btMatrix3x3& R) { init(R); }

	/**
	 * Create tb_angles from a matrix type. The matrix type must have its
	 * elements accessible as M[i][j]. This includes multidimensional C arrays.
	 */
	template<typename OtherType>
	tb_angles(OtherType R) {
		btMatrix3x3 btR;
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				btR[i][j] = R[i][j];
			}
		}
		init(btR);
	}
	
	/**
	 * Create tb_angles from a ROS geometry_msgs/Quaternion.
	 */
	tb_angles(const geometry_msgs::Quaternion& q) {
		btQuaternion btq(q.x,q.y,q.z,q.w);
		init(btMatrix3x3(btq));
	}

	/**
	 * Convert to btQuaternion.
	 */
	btQuaternion toQuaternion() const { btQuaternion q; toMatrix().getRotation(q); return q; }
	/**
	 * Convert to btTransform
	 */
	btTransform toTransform() const { btTransform T; T.setBasis(toMatrix()); return T; }
	/**
	 * Convert to btMatrix
	 */
	btMatrix3x3 toMatrix() const {
		btMatrix3x3 Ryaw(
				cos(yaw_rad), -sin(yaw_rad), 0,
				sin(yaw_rad),  cos(yaw_rad), 0,
				0,         0,        1);
		btMatrix3x3 Rpitch(
				cos(pitch_rad), 0, sin(pitch_rad),
				0,          1, 0,
				-sin(pitch_rad), 0, cos(pitch_rad));
		btMatrix3x3 Rroll(
				1,  0,          0,
				0,  cos(roll_rad), -sin(roll_rad),
				0,  sin(roll_rad),  cos(roll_rad));
		return Ryaw * Rpitch * Rroll;
	}
	/**
	 * Convert to ROS geometry_msgs/Quaternion
	 */
	geometry_msgs::Quaternion toMsg() const {
		btQuaternion btq = toQuaternion();
		geometry_msgs::Quaternion q;
		q.x = btq[0];
		q.y = btq[1];
		q.z = btq[2];
		q.w = btq[3];
		return q;
	}
	
	enum string_options { FIXED_WIDTH = 4, SHORT_NAMES = 8 };

	/**
	 * Get the string representation of this rotation.
	 * By default, the values are shown in degrees, without units.
	 * The options tb_angles::DEG and tb_angles::RAD can be used to display the
	 * values in degrees and radians, with units.
	 * tb.tostring():              values in degrees without units
	 * tb.tostring(tb_angles::DEG) values in degrees    with units
	 * tb.tostring(tb_angles::RAD) values in radians    with units
	 * tb.tostring(tb_angles::DEG & tb_angles::RAD) will print values in both degrees
	 *     and radians
	 *
	 * The option tb_angles::FIXED_WIDTH will cause the output to use fixed-width
	 * fields for the values, which can be useful if many values are being
	 * printed in succession.
	 *
	 * The option tb_angles::SHORT_NAMES causes the field names to be abbreviated
	 * to one letter, e.g., 'y' instead of 'yaw'
	 */
	std::string toString(int options = 0) const {
			char buf[120];
			const char* deg_fmt_fixed = "% 6.1f";
			const char* deg_fmt_var = "%.1f";
			const char* rad_fmt_fixed = "% 6.3f";
			const char* rad_fmt_var = "%.3f";

			bool deg = (options & DEG) || !(options & RAD);
			bool rad = options & RAD;

			std::string deg_fmt_str;
			std::string rad_fmt_str;
			if (options & FIXED_WIDTH) {
				deg_fmt_str = deg_fmt_fixed;
				rad_fmt_str = rad_fmt_fixed;
			} else {
				deg_fmt_str = deg_fmt_var;
				rad_fmt_str = rad_fmt_var;
			}

			std::string deg_str;
			std::string rad_str;
			if (deg) {
				if (! (options & DEG)) {
					deg_str = deg_fmt_str;
				} else {
					deg_str = deg_fmt_str + " deg";
				}
			} else {
				deg_str = "%.s";
			}
			if (rad) {
				rad_str = rad_fmt_str + " rad";
				if (deg) {
					rad_str = " (" + rad_str + ")";
				}
			} else {
				rad_str = "%.s";
			}

			char yaw_deg_str[20];   sprintf(yaw_deg_str,deg_str.c_str(),yaw_deg);
			char pitch_deg_str[20]; sprintf(pitch_deg_str,deg_str.c_str(),pitch_deg);
			char roll_deg_str[20];  sprintf(roll_deg_str,deg_str.c_str(),roll_deg);

			char yaw_rad_str[20];   sprintf(yaw_rad_str,rad_str.c_str(),yaw_rad);
			char pitch_rad_str[20]; sprintf(pitch_rad_str,rad_str.c_str(),pitch_rad);
			char roll_rad_str[20];  sprintf(roll_rad_str,rad_str.c_str(),roll_rad);

			char yaw_str[35];
			char pitch_str[35];
			char roll_str[35];

			sprintf(yaw_str,"%s%s",yaw_deg_str,yaw_rad_str);
			sprintf(pitch_str,"%s%s",pitch_deg_str,pitch_rad_str);
			sprintf(roll_str,"%s%s",roll_deg_str,roll_rad_str);

			std::string fmt_str1;
			if (options & SHORT_NAMES) {
				fmt_str1 = "[y:%s, p:%s, r:%s]";
			} else {
				fmt_str1 = "[yaw:%s, pitch:%s, roll:%s]";
			}

			sprintf(buf,fmt_str1.c_str(),yaw_str,pitch_str,roll_str);
			return std::string(buf);
		}

	/**
	 * Update the radian fields from the degree fields (e.g., yaw_rad from
	 * yaw_deg). Call this method after modifying one of the degree fields.
	 */
	void updateFromDegrees() {
		yaw_rad = yaw_deg * M_PI / 180.;
		pitch_rad = pitch_deg * M_PI / 180.;
		roll_rad = roll_deg * M_PI / 180.;
	}

	/**
	 * Update the degree fields from the radian fields (e.g., yaw_deg from
	 * yaw_rad). Call this method after modifying one of the radian fields.
	 */
	void updateFromRadians() {
		yaw_deg = yaw_deg * 180. / M_PI;
		pitch_deg = pitch_deg * 180. / M_PI;
		roll_deg = roll_deg * 180. / M_PI;
	}

private:
	void init(const btMatrix3x3& R) {
		yaw_rad = 0;
		pitch_rad = 0;
		roll_rad = 0;

		bool skip = false;
		if (fabs(R[0][1]-R[1][0]) < TB_ANGLES_CLOSE_ENOUGH && fabs(R[0][2]-R[2][0]) < TB_ANGLES_CLOSE_ENOUGH && fabs(R[1][2]-R[2][1]) < TB_ANGLES_CLOSE_ENOUGH) {
			//matrix is symmetric
			if (fabs(R[0][1]+R[1][0]) < TB_ANGLES_CLOSE_ENOUGH && fabs(R[0][2]+R[2][0]) < TB_ANGLES_CLOSE_ENOUGH && fabs(R[1][2]+R[2][1]) < TB_ANGLES_CLOSE_ENOUGH) {
				//diagonal
				if (R[0][0] > 0) {
					if (R[1][1] > 0) {
						skip = true;
					} else {
						roll_rad = M_PI;
					}
				} else if (R[1][1] > 0) {
					pitch_rad = M_PI;
				} else {
					yaw_rad = M_PI;
				}
				skip = true;
			}
		}

		if (!skip) {
			btVector3 vx = R * btVector3(1,0,0);
			btVector3 vy = R * btVector3(0,1,0);

			yaw_rad = atan2(vx.y(),vx.x());
			pitch_rad = atan2(-vx.z(), sqrt(vx.x()*vx.x() + vx.y()*vx.y()));

			btMatrix3x3 Ryaw(
						 cos(yaw_rad), -sin(yaw_rad), 0,
						 sin(yaw_rad),  cos(yaw_rad), 0,
						 0,         0,        1);
			btMatrix3x3 Rpitch(
					 cos(pitch_rad), 0, sin(pitch_rad),
					 0,          1, 0,
					-sin(pitch_rad), 0, cos(pitch_rad));
			btVector3 vyp = Ryaw * Rpitch * btVector3(0,1,0);
			btVector3 vzp = Ryaw * Rpitch * btVector3(0,0,1);

			float coeff = vzp.dot(vy) >= 0 ? 1 : -1;

			roll_rad = coeff * acos(vyp.dot(vy));
		}

		yaw_deg = yaw_rad * 180. / M_PI;
		pitch_deg = pitch_rad * 180. / M_PI;
		roll_deg = roll_rad * 180. / M_PI;
	}
};

} //namespace tfx

inline std::ostream& operator<<(std::ostream& o, const tfx::tb_angles& tb) {
	o << tb.toString();
	return o;
}

#endif
