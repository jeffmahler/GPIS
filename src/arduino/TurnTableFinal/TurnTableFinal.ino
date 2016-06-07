#include "PID_v1.h"

boolean pidControl = false;

double Target = 2;
float targetFilter = .2; // 1 is slow, 0 is no filter

double State = 2;
double potState = 2;
double Output = 2;
int potVals = 2;

float potFilter = .8; // 1 is slow, 0 is no filter

int sampleTime = 10;

float positionState = 2;

int pidTunings[3] = {600, 0, 0};
int maxBounds = 50;
int minBounds = 50;


double maxStates = 6.77;
double minStates = 0.48;

int valueOffsets = 1003;

double pi = 3.14159;

double potCalibrations =  (2 * pi) / (182 - 1003);

PID turntable(&State, &Output, &Target, pidTunings[0], pidTunings[1], pidTunings[2], DIRECT);

// Motor pins PWM
int motorDirPin = 10;
int motorDir = 1;
int motorPins =  11;
int PWM = {};

// Reset Mechanism Settings
int pwr = 240; //PWM
int resetPin = 7;
int resetMotorPins[] = {3, 5};
int resetPWM[] = {};
long resetStart = 0;
long resetTime = 2000;
long downTime = 5000;
boolean goingUp = false;
boolean reset = false;

// Potentiometer pins Analog IN
int potPins = A1;
int pots = 0;
long startTime = 0;
long now = millis();
long last = millis();


void setup() {
  // PWM analog outputs
  pinMode(motorPins, OUTPUT);
  pinMode(resetPin, INPUT);
  pinMode(resetMotorPins[0], OUTPUT);
  pinMode(resetMotorPins[1], OUTPUT);

  PWM = 0;
  pinMode(potPins, INPUT);

  Serial.begin(115200); // start serial for output
  Serial.setTimeout(200);

  turntable.SetMode(AUTOMATIC);
  turntable.SetOutputLimits(-minBounds, maxBounds);
  turntable.SetSampleTime(sampleTime);


  updateState();
  Target = State;
}


void loop() {
  now = millis();
//  if ((now - startTime) > 300) {
//    for (int i = 0; i < 12; i++) {
//      PWM = 0;
//      pidControl = false;
//      startTime = now;
//    }
  //}
  // reset code 
  if (reset == true){
    if (goingUp){  
      analogWrite(resetMotorPins[0],pwr);
      analogWrite(resetMotorPins[1],0); 
    }
    else {
      if (now > (resetStart + resetTime)){
        analogWrite(resetMotorPins[0],0);
        analogWrite(resetMotorPins[1],pwr); 
        if (now > (resetStart + resetTime + downTime)){
          analogWrite(resetMotorPins[0],0);
          analogWrite(resetMotorPins[1],0); 
          reset = false;
        }
      }
    }
  }

  if (digitalRead(resetPin) && goingUp){
    // STOP
    goingUp = false;
    resetStart = now;
    analogWrite(resetMotorPins[0],0);
    analogWrite(resetMotorPins[1],0);
  }


  updateState();

  if (pidControl){
    turntable.Compute();
  if (Output > 0) {
    motorDir = 0;
  }
  else {
    motorDir = 1;
  }
  PWM = abs(Output);
  }
  else{
    Target = State;
  }
  
  
  // software stops, checking if out of bounds
  if (State > maxStates && motorDir == 0) {
    PWM = 0;
    motorDir == 1;
  }
  else if (State < minStates && motorDir == 1) {
    PWM = 0;
    motorDir = 0;
  }
  // Write PWM to motors
  analogWrite(motorPins, PWM);
  digitalWrite(motorDirPin, motorDir);
}




void serialEvent() {
  startTime = millis();
  if (Serial.available() > 0) {
    // read the incoming byte:
    char command = Serial.read();

    // direct control
    if (command == 's') {
      pidControl = false;
      byte nums[2] = {};
      if (Serial.readBytes(nums, 2) == 2) {
        if (int(nums[0]) == 0) {
          motorDir = 1;
        }
        else {
          motorDir = 0;
        }
        PWM = abs(int(nums[0]) + int(nums[1]));
      }
      else {
          PWM = 0;    
      }
    }
    else if (command == 'r'){
      // RESET THE WHOLE SYSTEM
      reset = true;
      goingUp = true;
      resetStart = millis();
    }
    // reading a new target
    else if (command == 'a') {
      pidControl = true;
      int32_t bigNum;
      byte nums[4] = {};
      if (Serial.readBytes(nums, 4) == 4) {
        char a = nums[0];
        bigNum = a;
        bigNum = (bigNum << 8) | nums[1];
        bigNum = (bigNum << 8) | nums[2];
        bigNum = (bigNum << 8) | nums[3];
        Target = smooth(bigNum / 10000000.0, targetFilter, Target);
      }
    }

    else if (command == 'b') {
      Serial.println(State, 4);
    }
    else if (command == 'q') {
      Serial.println(Output);
    }
    else if (command == 'f'){
      //target = target + 50;
      analogWrite(resetMotorPins[0],0);
      analogWrite(resetMotorPins[1],155); 
    }
    else if (command == 's'){
      //target = target + 50;
      analogWrite(resetMotorPins[0],0);
      analogWrite(resetMotorPins[1],0); 
    }
    
  }
}


void updateState() {

  // Potentiometer State
  potVals = analogRead(potPins);
  potState = potCalibrations * (potVals - valueOffsets);

  // Smoothing filter for single pots
  State = smooth(potState, potFilter, State);

  // Target follows State during direct control to avoid a discontinuity when control is switched over
  if (!pidControl) {
    Target = State;
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
