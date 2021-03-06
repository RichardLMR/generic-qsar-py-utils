#########################################################################################################
#run_tests.py
#Runs all unit tests for the custom Python modules available as part of the genericQSARpyUtils project (see below)
# - i.e. ml_functions.py and ml_input_utils.py which you should have received along with this file.
#
#########
#Usage: #
#########
#(1) To run unit tests:
#python run_tests.py (or python <PATH TO run_tests.py>\run_tests.py)
#(2) To report the license information and then exit:
#python run_tests.py -l (or python <PATH TO run_tests.py>\run_tests.py -l)

# ##############################################
# #run_tests.py: Key documentation :Contents#
# ##############################################
# #1. Overview of this project.
# #2. IMPORTANT LEGAL ISSUES
# #<N.B.: Check this section ("IMPORTANT LEGAL ISSUES") to see whether - and how - you ARE ALLOWED TO use this code!>
# #<N.B.: Includes contact details.>
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
# #Defines two classes:
# #(i)descriptorsGenerator: This contains methods which can be used to interconvert between molecular file formats (e.g. SDF, SMILES,...), write the molecule ID to an SDF field, as well as calculate fingerprints presenting raw text codes for substructural features (e.g. extended connectivity fingerprints using jCompoundMapper or scaffold fragment fingerprints).
# #(ii)descriptorsFilesProcessor: This contains methods which can be used to prepare datasets in either CSV or svmlight format, including converting between these formats, based upon previously calculated fingerprints (expressed as a set of tab separated text strings for each instance) or numeric descriptors.
# #(2) ml_functions.py
# #Defines a set of functions which can be used to carry out univariate feature selection,cross-validation etc. for Machine Learning model input files in svmlight format.
# ###########################
# #2. IMPORTANT LEGAL ISSUES#
# ###########################
# Copyright Syngenta Limited 2013
#Copyright (c) 2013-2015 Liverpool John Moores University
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or (at
# your option) any later version.

# THIS PROGRAM IS MADE AVAILABLE FOR DISTRIBUTION WITHOUT ANY FORM OF WARRANTY TO THE 
# EXTENT PERMITTED BY APPLICABLE LAW.  THE COPYRIGHT HOLDER PROVIDES THE PROGRAM \"AS IS\" 
# WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT  
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
# PURPOSE. THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE PROGRAM LIES
# WITH THE USER.  SHOULD THE PROGRAM PROVE DEFECTIVE IN ANY WAY, THE USER ASSUMES THE
# COST OF ALL NECESSARY SERVICING, REPAIR OR CORRECTION. THE COPYRIGHT HOLDER IS NOT 
# RESPONSIBLE FOR ANY AMENDMENT, MODIFICATION OR OTHER ENHANCEMENT MADE TO THE PROGRAM 
# BY ANY USER WHO REDISTRIBUTES THE PROGRAM SO AMENDED, MODIFIED OR ENHANCED.

# IN NO EVENT UNLESS REQUIRED BY APPLICABLE LAW OR AGREED TO IN WRITING WILL THE 
# COPYRIGHT HOLDER BE LIABLE TO ANY USER FOR DAMAGES, INCLUDING ANY GENERAL, SPECIAL,
# INCIDENTAL OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OR INABILITY TO USE THE
# PROGRAM (INCLUDING BUT NOT LIMITED TO LOSS OF DATA OR DATA BEING RENDERED INACCURATE
# OR LOSSES SUSTAINED BY THE USER OR THIRD PARTIES OR A FAILURE OF THE PROGRAM TO 
# OPERATE WITH ANY OTHER PROGRAMS), EVEN IF SUCH HOLDER HAS BEEN ADVISED OF THE 
# POSSIBILITY OF SUCH DAMAGES.

# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

# ####################
# See also: http://www.gnu.org/licenses/ (last accessed 14/01/2013)

# Contact: 
# 1. richard.marchese_robinson@syngenta.com
# or if this fails
# 2. rmarcheserobinson@gmail.com
# #####################
#########################################################################################################
'''
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

'''
import sys,getopt,os,re

dir_of_this_script = "\\".join(os.path.abspath(__file__).split('\\')[:-1])

def report_all_files_and_dirs_in_dir_of_this_script_prior_to_running_unit_tests():
	return [r'%s\%s' % (dir_of_this_script,file_or_dir_name) for file_or_dir_name in os.listdir(r'%s' % dir_of_this_script)]

def cleanUp(all_files_and_dirs_in_dir_of_this_script_prior_to_running_unit_tests): 
	import glob
	#some unit tests may actually write intermediate temporary files to dir_of_this_script - hence make sure that these have all been deleted after all unit tests have been carried out!
	
	unit_test_temp_files_written_to_dir_of_this_script = [file_name for file_name in glob.glob(r'%s\*' % dir_of_this_script) if not file_name in all_files_and_dirs_in_dir_of_this_script_prior_to_running_unit_tests]
	
	for UNIT_TEST_TEMP_FILE_WRITTEN_TO_DIR_OF_THIS_SCRIPT in unit_test_temp_files_written_to_dir_of_this_script:
		print '='*50
		print 'Removing this temporary file written to the directory of run_tests.py: ', UNIT_TEST_TEMP_FILE_WRITTEN_TO_DIR_OF_THIS_SCRIPT
		os.remove(UNIT_TEST_TEMP_FILE_WRITTEN_TO_DIR_OF_THIS_SCRIPT)
		print '='*50


def run_all_ml_functions_and_ml_input_utils_unit_tests():
	import unittest
	import glob
	
	all_files_and_dirs_in_dir_of_this_script_prior_to_running_unit_tests = report_all_files_and_dirs_in_dir_of_this_script_prior_to_running_unit_tests()
	
	CWD = os.getcwd()
	os.chdir(dir_of_this_script)
	
	test_dirs = [file_or_dir_name for file_or_dir_name in os.listdir(r'%s' % os.getcwd()) if re.match('(test_[0-9]+$)',file_or_dir_name)]
	#print 'test_dirs = ', test_dirs
	#sys.exit(1)
	########################################
	#The following code was written based on the answer provided by Stephen Cagle[http://stackoverflow.com/users/21317/stephen-cagle]: http://stackoverflow.com/questions/1732438/how-to-run-all-python-unit-tests-in-a-directory
	test_file_names = []
	
	for TEST_DIR in test_dirs:
		sys.path.append(r'%s\%s' % (dir_of_this_script,TEST_DIR))
		os.chdir(TEST_DIR)
		test_file_names += glob.glob('test_*.py')
		os.chdir("..")
	del test_dirs
	del TEST_DIR
	
	test_module_names = [name[0:len(name)-3] for name in test_file_names]
	del name
	loadedTests = [unittest.defaultTestLoader.loadTestsFromName(name) for name in test_module_names]
	del name
	testsReady2Run = unittest.TestSuite(loadedTests)
	unittest.TextTestRunner().run(testsReady2Run)
	########################################
	os.chdir(CWD)
	
	cleanUp(all_files_and_dirs_in_dir_of_this_script_prior_to_running_unit_tests)


def main():
	
	opts,args = getopt.getopt(sys.argv[1:],'l',['license'])
	
	for o,v in opts:
		if '-l' == o:
			print __doc__
			sys.exit()
	try:
		del opts,args,o,v
	except UnboundLocalError:
		pass
	
	#print 'Hello!'
	#print "\\".join(os.path.abspath(__file__).split('\\')[:-1])
	#print 'Hello again!'
	
	run_all_ml_functions_and_ml_input_utils_unit_tests()
	
	return 0

if __name__ == '__main__':
	sys.exit(main())

 