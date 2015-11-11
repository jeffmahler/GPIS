How to Control Robot w/ Xbox Controller:

Pre-reqs:
Make sure you have python 2.7, and have the proper dependencies installed (see the google doc about dex installation for more details). 
To run the files in this folder, you really only need:

-Numpy
-Pygame

Steps:

0. Connect usb from the Zeke robot to your computer. Also Connect Xbox Controller to computer.
1. Figure out which Comm port corresponds to the specific USB port, and modify ZekeScript accordingly.
2. Run ZekeScript.py in interactive mode
3. Call kinematicControl() 
4. Now you can directly control Zeke with the xbox controls. 