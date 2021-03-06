#####################
#README.txt:Contents#
#####################
#1. Overview of this project.
#2. IMPORTANT LEGAL ISSUES
#<N.B.: Check this section ("IMPORTANT LEGAL ISSUES") to see whether - and how - you ARE ALLOWED TO use this code!>
#<N.B.: Includes contact details.>
#3. Getting started
#<N.B.: Check this section ("Getting started") to see whether you CAN use this code!>
#4. Using the code
#
# ##############################
# #1. Overview of this project.#
# ##############################
# #Project name: genericQSARpyUtils
# #Purpose of this project: To provide a set of Python functions
# #(or classes with associated methods) that can be used to perform a variety of tasks
# #which are relevant to generating input files, from cheminformatics datasets, which can be used to build and
# #validate QSAR models (generated using Machine Learning methods implemented in other software packages)
# #on such datasets.
# #To this end, two Python modules are currently provided.
# #(1) ml_input_utils.py 
# #Defines the following class:
# #descriptorsFilesProcessor: This contains methods which can be used to prepare datasets in either CSV or svmlight format, including converting between these formats, based upon previously calculated fingerprints (expressed as a set of tab separated text strings for each instance) or numeric descriptors.
# #(2) ml_functions.py
# #Defines a set of functions which can be used to carry out univariate feature selection,cross-validation etc. for Machine Learning model input files in svmlight format.


###########################
#2. IMPORTANT LEGAL ISSUES#
###########################
Copyright Syngenta Limited 2013
#Copyright (c) 2013-2015 Liverpool John Moores University
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or (at
your option) any later version.

THIS PROGRAM IS MADE AVAILABLE FOR DISTRIBUTION WITHOUT ANY FORM OF WARRANTY TO THE 
EXTENT PERMITTED BY APPLICABLE LAW.  THE COPYRIGHT HOLDER PROVIDES THE PROGRAM \"AS IS\" 
WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT  
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
PURPOSE. THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE PROGRAM LIES
WITH THE USER.  SHOULD THE PROGRAM PROVE DEFECTIVE IN ANY WAY, THE USER ASSUMES THE
COST OF ALL NECESSARY SERVICING, REPAIR OR CORRECTION. THE COPYRIGHT HOLDER IS NOT 
RESPONSIBLE FOR ANY AMENDMENT, MODIFICATION OR OTHER ENHANCEMENT MADE TO THE PROGRAM 
BY ANY USER WHO REDISTRIBUTES THE PROGRAM SO AMENDED, MODIFIED OR ENHANCED.

IN NO EVENT UNLESS REQUIRED BY APPLICABLE LAW OR AGREED TO IN WRITING WILL THE 
COPYRIGHT HOLDER BE LIABLE TO ANY USER FOR DAMAGES, INCLUDING ANY GENERAL, SPECIAL,
INCIDENTAL OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OR INABILITY TO USE THE
PROGRAM (INCLUDING BUT NOT LIMITED TO LOSS OF DATA OR DATA BEING RENDERED INACCURATE
OR LOSSES SUSTAINED BY THE USER OR THIRD PARTIES OR A FAILURE OF THE PROGRAM TO 
OPERATE WITH ANY OTHER PROGRAMS), EVEN IF SUCH HOLDER HAS BEEN ADVISED OF THE 
POSSIBILITY OF SUCH DAMAGES.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

####################
See also: http://www.gnu.org/licenses/ (last accessed 14/01/2013)

Contact: 
1. R.L.MarcheseRobinson@ljmu.ac.uk
or if this fails
2. rmarcheserobinson@gmail.com
#####################

####################
#3. Getting started#
####################

N.B.: You need the Windows operating system running on your machine. 
See "Prerequistes" point "0" for further details. 

##############
#Installation#
##############

1. Create a "main installation directory" anywhere on your machine that scripts can be executed from.
This can be called anything you like -e.g. "C:\mainInstall".

2.This file (i.e. README.txt) should have been automatically extracted, when you downloaded this code,
into a directory called "code". Place this directory inside your main installation directory - e.g. such that
"C:\mainInstall" will contain a folder called "code", with README.txt now residing in "C:\mainInstall\code".

###############
#Prerequisites#
###############
0. To date, this code has only been tested on a machine running Windows 7. Some modifications may be required to enable it to be used on a machine running a unix-type operating system - e.g. file name manipulation and specification of path names.
1. Python interpreter. http://python.org/ Python version >= 2.5 (at least) is required. Python version 2.7.3 is recommended since all code development employed this version of Python.
2. NumPy Python module. All code development used the module installed by running numpy-MKL-1.6.2.win32-py2.7.exe obtained from http://www.lfd.uci.edu/~gohlke/pythonlibs/#scikit-learn (accessed 22/08/2012).
3. SciPy Python module. All code development used the module installed by running scipy-0.10.1.win32-py2.7.exe obtained from http://www.lfd.uci.edu/~gohlke/pythonlibs/#scikit-learn (accessed 22/08/2012).
4. MatPlotLib Python module. All code development used the module installed by running matplotlib-1.1.1.win32-py2.7.exe obtained from http://www.lfd.uci.edu/~gohlke/pythonlibs/#scikit-learn (accessed 22/08/2012).
5. scikit-learn Python module. All code development used the module installed by running scikit-learn-0.13.win32-py2.7.exe obtained from http://www.lfd.uci.edu/~gohlke/pythonlibs/#scikit-learn (accessed 23/03/2013).
6. OpenBabel. http://openbabel.org/wiki/Main_Page All code development used the version installed via running OpenBabel2.3.0a_Windows_Installer.exe.
7. Pybel Python module  for interfacing with OpenBabel. http://openbabel.org/wiki/Python All code development used the version installed via running openbabel-python-1.6.py27.exe.  
8. ml_functions.py, ml_input_utils.py should be located in a subdirectory of the main installation directory of this project called "code".
9. run_tests.py must be located in a subdirectory of the main installation directory called "code\tests" , with all individual unit tests located in subdirectories named test_<x>, in files called test_<x>.py, where x = 1,2,3... etc. along with their requisite input and output files) designed to test the functionality of the ml_functions.py and ml_input_utils.py modules.

####################
#4. Using the code #
####################

1. To use these modules within a Python script, insert the following lines of code into your script.


import sys
sys.path.append(<ABSOLUTE PATH OF THE "MAIN INSTALLATION DIRECTORY">\code) 
#This last line allows you to import from the ml_input_utils and ml_functions modules

####Depending on what functionality you wanted to use, this line would be followed by something like: 
import ml_input_utils
#AND/OR
import ml_functions
#OR e.g.
from ml_input_utils import descriptorsFilesProcessor

2. To test the code, try running:
python <ABSOLUTE PATH OF THE "MAIN INSTALLATION DIRECTORY">\code\tests\run_tests.py
Search within the output for "Ran <NUMBER OF TESTS> tests in <NUMBER OF SECONDS>s". 

If this message is followed by "OK", all tests passed.
BUT if this is followed by a message indicating failures and/or errors, search in the preceding output
for the associated failures and errors to see what has gone wrong!

3. For examples of how you might use the code to achieve specific tasks, as well as a confirmation 
of which parts of the code have associated unit tests, look at the contents of the test_<x>.py files found in these
directories:
<ABSOLUTE PATH OF THE "MAIN INSTALLATION DIRECTORY">\code\tests\test_<x>\



 