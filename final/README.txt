This program is a 

In order to run this program, you will likely have to download the packages tensorflow and keras.
-if you are using PyCharm, you can install them by hovering over the library names at the top of RNN.py, and the IDE will do it automatically

To demonstrate the program, run RNN.py.
-this program takes some time to run successfully (5 or more minutes)
-only when an error is reached has the program stopped running (thus far 
-there are 2 major pauses, first when it is calculating the training data, and second when it is calculating on the testing data.

ERROR:
one error appears to be due to the version of Keras your machine installs, and as such we have been unable to account for it
-If the program has an error on first run, on line 83, replace the parameter 'weight_decay' with 'decay'. 