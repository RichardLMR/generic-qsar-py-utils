#########################################################################################################
# test_10.py
# Implements unit tests for the genericQSARpyUtils project (see below).
#
# ########################################
# #test_10.py: Key documentation :Contents#
# ########################################
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
# #Defines the following class:
# #descriptorsFilesProcessor: This contains methods which can be used to prepare datasets in either CSV or svmlight format, including converting between these formats, based upon previously calculated fingerprints (expressed as a set of tab separated text strings for each instance) or numeric descriptors.
# #(2) ml_functions.py
# #Defines a set of functions which can be used to carry out univariate feature selection,cross-validation etc. for Machine Learning model input files in svmlight format.
# ###########################
# #2. IMPORTANT LEGAL ISSUES#
# ###########################
# Copyright Syngenta Limited 2013
#Copyright (c) 2013-2016 Liverpool John Moores University
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
import sys,re,os,glob
project_name = 'genericQSARpyUtils'
current_dir = r'%s' % os.path.dirname(os.path.abspath(__file__))
project_modules_to_test_dir = os.path.split(os.path.split(current_dir)[0])[0]
sys.path.append(project_modules_to_test_dir)


import unittest
class test_10(unittest.TestCase):
	
	def clean_up_if_all_checks_passed(self,current_dir,specific_files_not_to_delete):
		all_files_to_delete = [file_name for file_name in glob.glob(r'%s\*' % current_dir) if not re.search('(.\py$)',file_name) and not file_name in specific_files_not_to_delete]
		
		for FILE_TO_DELETE in all_files_to_delete:
			os.remove(FILE_TO_DELETE)
			assert not os.path.exists(FILE_TO_DELETE), " This still exists: \n %s" % FILE_TO_DELETE
			print 'Removed this temporary file: ', FILE_TO_DELETE
	
	def compareExpectedAndActualFile(self,expected_file,actual_file):
		
		print '-'*50
		print 'Comparing: '
		print expected_file
		print 'to:'
		print actual_file
		print '-'*50
		
		file2Contents = {}
		
		for file_name in [expected_file,actual_file]:
			f_in = open(file_name)
			try:
				file2Contents[file_name] = ''.join([re.sub(r'\r|\n','',LINE) for LINE in f_in.readlines()])
				del LINE
			finally:
				f_in.close()
				del f_in
		del file_name
		
		assert file2Contents[expected_file] == file2Contents[actual_file], "expected_file=%s,actual_file=%s: their contents do not match!" % (expected_file,actual_file)
	
	def compareAllExpectedAndActualFiles(self,current_dir):
		
		all_expected_files = glob.glob(r'%s\*_Expected*' % current_dir)
		assert not 0 == len(all_expected_files)
		
		for expected_file in all_expected_files:
			actual_file = re.sub('(_Expected)','',expected_file)
			assert not actual_file == expected_file, "expected_file=%s. ditto actual_file???" % expected_file
			self.compareExpectedAndActualFile(expected_file,actual_file)
	
	def test_10_predefined_split_svmlight_file(self):
		##############################
		print 'Running unittests for this project: ', project_name
		print 'Running this unittest: ', self._testMethodName
		##################################
		
		do_not_to_delete = glob.glob(r'%s\*' % current_dir)
		
		from ml_functions import predefined_split_svmlight_file
		
		subset2svmlightFile = predefined_split_svmlight_file(svmlight_file=r'%s\toy10_complete_svmlight_file.txt' % current_dir,train_IDs_file=r'%s\toy10_train_IDs_file.txt' % current_dir,test_IDs_file=r'%s\toy10_test_IDs_file.txt' % current_dir,output_label='_toy.out.label')
		
		test_10_return_var_contents_record  = r'%s\Names_of_svmlight_train_test_files_test_10.txt' % current_dir
		f_out = open(test_10_return_var_contents_record,'w')
		try:
			f_out.write('svmlight_train_file = %s\n'% os.path.relpath(subset2svmlightFile['TRAIN'],start=current_dir))
			f_out.write('svmlight_test_file = %s\n'% os.path.relpath(subset2svmlightFile['TEST'],start=current_dir))
		finally:
			f_out.close()
			del f_out
		
		self.compareAllExpectedAndActualFiles(current_dir)
		
		self.clean_up_if_all_checks_passed(current_dir,specific_files_not_to_delete=do_not_to_delete)
