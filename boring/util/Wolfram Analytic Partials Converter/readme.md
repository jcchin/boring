Written 10/26/20 K. Look

This is a test script that converts python equations into wolfram alpha queries, then converts the wolfram queries back into python code. 
Purpose is for perfoming complicated analytical partial derivates for component files

This script is only partially complete, currently has issues with resolving variable names that contain part of other variable names (e.g. "a" in alpha gets converted to "alphalpha")

Future work should focus on using Sympy or other python analytic solver to remove black box dependency on wolfram alpha, and also manual work of sending queries and receiving responses. 