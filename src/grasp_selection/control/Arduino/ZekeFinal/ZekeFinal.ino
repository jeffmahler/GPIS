#include <PID_v1.h>

boolean pidControl = false;

double Target[6] = {0};
float targetFilter = .3;

double encState[] = {2, 2, 2};
double State[6] = {1, 1, 1, 1, 1, 1};
double potState[6] = {};
double Output[6] = {};

float potFilter = .75; // 1 is slow, 0 is no filter
float compPotFilter = .65;
float compFilter = .98; // favors encoder changes

float positionCutoff[6] = {.02, .002, .002};

int sampleTime = 20;

float positionState[6] = {};

int pidTunings[6][3] =

{ {900, 0, 30},
  {20000, 20000, 0},
  {8000, 0, 300},
  {500, 0, 40},
  {2000, 0, 200},
  {700, 0, 20}
};

// FILTERS
// the velocityCutoff is the level where the state estimation switches from the complimentary filter to just the encoders
// The idea is the encoders have drift, but to only fix that drift while the robot is moving, so for small slow
// movements, the robot has the resolution of the encoders, and is not fixed to the resolution of the potentiometers


int maxBounds[6] = {180, 200, 155, 155, 255, 120};
int minBounds[6] = {180, 100, 155, 155, 255, 120};

double maxStates[6] = {6.58991792126, .3, 0.3, 6.61606370861, 0.0348490572685, 6.83086639225};
double minStates[6] = {0, .02, .02, 0.183086039735, -.01, 0.197775641646};

int valueOffsets[6] = {728, 0, 81, 106, 819, 23};

double pi = 3.14159;

// encoders 1-3, encoderVelocity 1-3,turntableVelocity, potentiometers 1-6, currents 1-4

double encoderCalibrations[3] = { 2 * pi / (131615 - 176669),
                                  13.6875 / (224808) / 39.3701,
                                  .83 * (13.6875 - .75) / (-65461) / 39.3701
                                };

double potCalibrations[6] = {(-2 * pi / (688 - 53)),
                             13.6875 / (347) / 39.3701,
                             ((13.6875 - .75) / (416 - 82)) / 39.3701,
                             (-2 * pi) / (135 - 890),
                             (7.0 / 8 - 2.25) / (823 - 363) / 39.3701,
                             (-2 * pi) / (77 - 903),
                            };


PID rot(&State[0], &Output[0], &Target[0], pidTunings[0][0], pidTunings[0][1], pidTunings[0][2], DIRECT);
PID elv(&State[1], &Output[1], &Target[1], pidTunings[1][0], pidTunings[1][1], pidTunings[1][2], DIRECT);
PID ext(&State[2], &Output[2], &Target[2], pidTunings[2][0], pidTunings[2][1], pidTunings[1][2], DIRECT);
PID wrist(&State[3], &Output[3], &Target[3], pidTunings[3][0], pidTunings[3][1], pidTunings[1][2], DIRECT);
PID grip(&State[4], &Output[4], &Target[4], pidTunings[4][0], pidTunings[4][1], pidTunings[1][2], DIRECT);
PID turntable(&State[5], &Output[5], &Target[5], pidTunings[5][0], pidTunings[5][1], pidTunings[1][2], DIRECT);

PID controllers[6] = {rot, elv, ext, wrist, grip, turntable};


#include <Encoder.h>

// Motor pins PWM
// Rotation Elevation Extension Wrist Grip
int motorPins[12] = { 7, 6,    13, 12,  4, 5,  9, 8,  11, 10, 44, 46};
int PWM[12] = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
// Motor Enable pins Digital IO
// Rotation, Elevation, Extension, Wrist, Grip
int enablePins[6] = {34, 36, 38, 42, 40, 48};
int enables[6] = {1, 1, 1, 1, 1, 1}; //set all to high

// Potentiometer pins Analog IN
// Rotation, Elevation, Extension, Wrist, Grip, Turntable
int potPins[6] = {A3, A5, A4, A6, A7, A2};
int pots[6] = {2, 2, 2, 2, 2, 2};

// Current Pins Analog IN
// Rotation, Elevation, Extension, GRIP NOT WRIST
int currentPins[4] = {A8, A9, A10, A11};
int currents[4];

//Declare Encoders With Interrupt Pins
Encoder enc1(18, 20); // Extension
Encoder enc2(19, 21); // Elevation
Encoder enc3(2, 3); // Rotation
long startTime = millis();
long newEnc[3] = {};
long oldEnc[3] = {};




void setup() {

  analogReference(EXTERNAL); // DO NOT REMOVE THIS

  // PWM analog outputs
  for (int i = 0; i < 12; i++) {
    pinMode(motorPins[i], OUTPUT);
  }

  for (int i = 0; i < 12; i++) {
    PWM[i] = 0;
  }

  // Digital Enable Outputs
  for (int i = 0; i < 6; i++) {
    pinMode(enablePins[i], OUTPUT);
  }

  // Analog In Potentiometers
  for (int i = 0; i < 6; i++) {
    pinMode(potPins[i], INPUT);
  }

  // Analog In Current Values
  for (int i = 0; i < 4; i++) {
    pinMode(currentPins[i], OUTPUT);
  }

  // Make all motors enabled all the time
  for (int i = 0; i < 6; i++) {
    digitalWrite(enablePins[i], HIGH);
  }

  Serial.begin(115200); // start serial for output
  Serial.setTimeout(200);

  for (int i = 0; i < 6; i++) {
    controllers[i].SetMode(AUTOMATIC);
    controllers[i].SetOutputLimits(-minBounds[i], maxBounds[i]);
    controllers[i].SetSampleTime(sampleTime);
  }


}


void loop() {
  if ((millis() - startTime) > 60) {
    for (int i = 0; i < 12; i++) {
      PWM[i] = 0;
      pidControl = false;
      startTime = millis();
    }
  }

  updateState();

  if (pidControl) {
    for (int i = 0; i < 6; i++) {
      controllers[i].Compute();
      if (Output[i] > 0) {
        PWM[i * 2] = Output[i];
        PWM[i * 2 + 1] = 0;
      }
      else {
        PWM[i * 2] = 0;
        PWM[i * 2 + 1] = abs(Output[i]);
      }
    }
  }


  // software stops, checking if out of bounds
  for (int i = 0; i < 6; i++) {
    if (State[i] > maxStates[i]) {
      PWM[i * 2] = 0;
    }
    else if (State[i] < minStates[i]) {
      PWM[i * 2 + 1] = 0;
    }
  }

  // Write PWM to motors
  for (int i = 0; i < 6; i++) {
    analogWrite(motorPins[i * 2], PWM[i * 2]);
    analogWrite(motorPins[i * 2 + 1], PWM[i * 2 + 1]);
  }
}




void serialEvent() {
  startTime = millis();
  if (Serial.available() > 0) {
    // read the incoming byte:
    char command = Serial.read();

    // direct control
    if (command == 's') {
      pidControl = false;
      byte nums[12] = {};
      if (Serial.readBytes(nums, 12) == 12) {
        for (int i = 0; i < 12; i++) {
          PWM[i] = int(nums[i]);
        }
      }
      else {
        for (int i = 0; i < 12; i++) {
          PWM[i] = 0;
        }
      }
    }

    // reading a new target
    else if (command == 'a') {
      pidControl = true;
      for (int i = 0; i < 6; i++) {
        int32_t bigNum;
        byte nums[4] = {};
        if (Serial.readBytes(nums, 4) == 4) {
          char a = nums[0];
          bigNum = a;
          bigNum = (bigNum << 8) | nums[1];
          bigNum = (bigNum << 8) | nums[2];
          bigNum = (bigNum << 8) | nums[3];
          Target[i] = smooth(bigNum / 10000000.0, targetFilter, Target[i]);
        }
      }
    }

    else if (command == 'b') {
      for (int i = 0; i < 6; i++) {
        Serial.println(State[i], 5);
      }
      Serial.flush();
    }
    else if (command == 'q') {
      Serial.println(newEnc[0]);
    }
  }
}

void updateState() {

  // Changes in encoder values

  // (I tried indexing from an array of encoder objects and it bugged out, so I do it this way sadly)
  newEnc[0] = enc3.read();
  newEnc[1] = enc2.read();
  newEnc[2] = enc1.read();
  for (int i = 0; i < 3; i++) {
    encState[i] = encoderCalibrations[i] * (newEnc[i] - oldEnc[i]);
    oldEnc[i] = newEnc[i];
  }

  encState[2] = encState[2] - encState[1]; // coupled motion

  // Current Measurements From Motors
  for (int i = 0; i < 4; i++) {
    currents[i] = analogRead(currentPins[i]);
  }

  // Potentiometer State
  for (int i = 0; i < 6; i++) {
    potState[i] = potCalibrations[i] * (analogRead(potPins[i]) - valueOffsets[i]);
  }
  potState[2] = potState[2] - potState[1]; // coupled motion


  // UPDATE STATE

  // Complimentary filter for encoders
  for (int i = 0; i < 3; i++) {
    float smoothedPot = smooth(potState[i], compPotFilter, State[i]);
    if (abs(State[i] - smoothedPot) < positionCutoff[i]) {
      State[i] = State[i] + encState[i];
    }
    else {
      State[i] = compFilter * (State[i] + encState[i]) + (1 - compFilter) * smoothedPot;
    }
  }

  // Smoothing filter for single pots
  for (int i = 3; i < 6; i++) {
    State[i] = smooth(potState[i], potFilter, State[i]);
  }

  // Target follows State during direct control to avoid a discontinuity when control is switched over
  if (!pidControl) {
    for (int i = 0; i < 6; i++) {
      Target[i] = State[i];
    }
  }
}

float smooth(float data, float filterVal, float smoothedVal) {
  if (filterVal > 1) {     // check to make sure param's are within range
    filterVal = .99;
  }
  else if (filterVal <= 0) {
    filterVal = 0;
  }
  smoothedVal = (data * (1 - filterVal)) + (smoothedVal  *  filterVal);

  return (float)smoothedVal;
}
