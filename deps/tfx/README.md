tfx aims to make working with transforms and poses much simpler. It is provided
as a ROS package, but the `tfx` python module can be imported without ROS
installed; ROS-related features will be simply disabled.

Modules
=======

The `tfx` module provides convenience classes for working with transformations.
The main classes are `tb_angles`, which represents rotations as easily-understood
Tait-Bryan angles (aircraft angles), and the classes from the `canonical` module.

Additionally, if ROS is available, importing the `tfx` module installs a singleton
factory on `tf.TransformListener` which frees up code from having to pass around
a listener, since only one instance can exist in a node. This singleton can be 
accessed through the function `tfx.TransformListener()`, providing a drop-in 
replacement in existing code. The module also provides convenience functions
for commonly-used operations on `TransformListener`.

tb_angles
---------

`tb_angles` exists to make it easier to view and create rotations. Most rotation
representations, such as quaternions, rotation matrices, and even axis-angle
are hard to visualize directly. However, Tait-Bryan angles, also known as 
aircraft angles, are very easy to visualize. It is an intrinsic (moving-axis)
yaw-pitch-roll (zyx) rotation. They are very similar to Euler angles, and in 
fact are often referred to as Euler angles. However, the term Tait-Bryan angles
is used here to clearly distinguish them from other forms of Euler angles.
Further documentation is available in the class doc.

canonical
---------

The `canonical` module contains several classes that have the purpose of making
it easier to convert between formats. There are five main classes:

* `CanonicalStamp` (timestamps)
* `CanonicalDuration` (time duration)
* `CanonicalPoint` (3D point or vector with optional frame and timestamp)
* `CanonicalRotation` (3D rotation with optional frame and timestamp)
* `CanonicalTransform` (3D transform or pose)

Each class is capable of being created from a variety of formats, and each has
methods for converting back to these formats.

The classes should not be created directly. The following functions, which are
imported into the `tfx` namespace, provide None-safe creation, error checking,
and, in the case of `transform()` and `pose()`, the ability to convert ROS message
types that contain arrays (`PoseArray` and `tfMessage`) into lists of 
`CanonicalTransforms`:

* `stamp()`
* `duration()`
* `time()`: allows conversion to stamp or duration based on input
* `point()` or `vector()`
* `rotation()` or `quaternion()`: equivalent, both provided for convenience
* `transform()` and `pose()`: nearly equivalent, see below.

Two methods `rotation_tb` and `rotation_euler` are provided for creating
`CanonicalRotation`s from Tait-Bryan and Euler angles, respectively.

Methods for creating identity transforms, poses, and rotations, and random
points, rotations, transforms and poses are also provided.

`CanonicalTransform` represents both transforms and poses. The transform from
frame A to frame B is equal to the pose of frame A's axes in frame B. This
equivalence is used in `CanonicalTransform`. Whether the object is a pose or a
transform is stored in the object's `pose` attribute.
There are two frame fields on `CanonicalTransform`, each of which has multiple
aliases:
`parent` and `frame`:
    The destination frame of the transform or the frame of the pose.
`child` and `name`:
    The source frame of the transform or a name for the pose.

Mathematical operations on canonical objects use operator overloading.
Multiplying two transforms T1 and T2 is T1 * T2. To this end, `CanonicalPoint`,
`CanonicalRotation`, and `CanonicalTransform` are subclasses of numpy.matrix, which
has the same semantics. If a numpy array is needed, the `array` property will
return a view of the data as an array.
Mathematical operations are auto-converting; for example, given a
`CanonicalTransform`, right-multiplying a (non-canonical) point will succeed if
the point is in any format that can be recognized by `CanonicalPoint`.
Left-multiplication of a non-canonical object will autoconvert if the object
does not support multiplication with canonical objects; because the left object
has priority in applying the multiplication, conversion cannot be guaranteed in
all cases.
The only exception is multiplication of a transform with a sequence of length 4
with elements 0,0,0,1: this could be an identity rotation as a quaternion, or
the point (0,0,0) as a 4-vector. In this case, an exception will be raised.

Operations on canonical objects are frame-aware. If the objects
have frames defined, and the frames conflict, an exception will be raised.
If a transform from frame A to frame B is applied by left-multiplication to a
point or rotation in frame B, the transform will automatically apply its
inverse and return the result in frame A.

The `copy()` methods on the objects allow keyword arguments for replacing fields
on the copy; for example, for a `CanonicalPoint` `p`, `p.copy(x=0,stamp='now')` will
return a copy with the same frame (if any), y and z values, but with x set to 0
and the timestamp set to the current time.

Conversion to ROS for `CanonicalStamp` and `CanonicalDuration` is provided by the
`ros` attribute, which returns a `rospy.Time` or `rospy.Duration`, respectively.
On `CanonicalPoint`, `CanonicalRotation`, and `CanonicalTransform`, conversion to ROS
messages is provided by a generator object available at the `msg` attribute.
This generator object has methods named for each of the messages (despite being
methods, the initial letter is capitalized). The generators have two methods in
common, named `Header` and `stamp`. Each of these has two keyword arguments,
`default_stamp` and `stamp_now`. Setting `default_stamp` to `True` indicates that
if no stamp is set, the current time should be used. Setting `stamp_now` to `True`
overrides `default_stamp` if it is set, and always sets the stamp to be the 
current time (without changing the stamp on the object). The methods for 
creating stamped messages also take these keyword arguments. The `tfMessage`
method on the message generator for `CanonicalTransform` is slightly different; 
see its documentation for details.

Scripts
=======

tfx provides several convenience scripts:

tf_echo
-------

This functions similar to tf_echo in the tf package, but prints in the format
used by `CanonicalTransform.tostring()` by default (i.e., using Tait-Bryan 
angles for the rotation), so the rotation should be easier to interpret.

topic_echo
----------

This script functions similar to `rostopic echo`, but works on `Pose`,
`PoseStamped`, `Transform`, and `TransformStamped` messages, printing them
in the format used by `CanonicalTransform.tostring()` by default (i.e., using
Tait-Bryan angles for the rotation), so the rotation should be easier to 
interpret.

transform/pose publishing
-------------------------

Publishing a transform or pose can be accomplished using one of three scripts,
which all use the same underlying code:

* pose_publisher, publishing pose by default
* tf_publisher, publising transform by default
* publisher, which can publish either transform or pose (for advanced usage)

The transform/pose published by these scripts can be static, dynamically
adjusted using keyboard input, or even based on a separate pose/transform topic.
See the documentation for options and examples.
