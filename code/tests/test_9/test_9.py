#########################################################################################################
# test_9.py
# Implements unit tests for the genericQSARpyUtils project (see below).
#
# ########################################
# #test_9.py: Key documentation :Contents#
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
#Copyright (c) Liverpool John Moores University 2014 - 2016
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
project_modules_to_test_dir = "\\".join(os.path.abspath(__file__).split('\\')[:-3])
sys.path.append(project_modules_to_test_dir)


import unittest
class test_9(unittest.TestCase):
	
	################################################################
	#========================================================
	#Description of testing carried out by this test class.
	#========================================================
	#Check ml_functions.predefined_split_svmlight_file(...)
	################################################################
	
	
	def clean_up_if_all_checks_passed(self,specific_files_not_to_delete):
		all_files_to_delete = [file_name for file_name in glob.glob(r'%s\*' % "\\".join(os.path.abspath(__file__).split('\\')[:-1])) if not re.search('(.\py$)',file_name) and not file_name in specific_files_not_to_delete]
		
		for FILE_TO_DELETE in all_files_to_delete:
			os.remove(FILE_TO_DELETE)
			assert not os.path.exists(FILE_TO_DELETE), " This still exists: \n %s" % FILE_TO_DELETE
			print 'Removed this temporary file: ', FILE_TO_DELETE
	
	def compareOriginalAndNewFiles(self,orig_file,new_file):
		
		print '-'*50
		print 'Comparing: '
		print orig_file
		print 'to:'
		print new_file
		print '-'*50
		
		file2Contents = {}
		
		for file_name in [orig_file,new_file]:
			f_in = open(file_name)
			try:
				file2Contents[file_name] = ''.join([re.sub(r'\r|\n','<EOL>',LINE) for LINE in f_in.readlines()])
				del LINE
			finally:
				f_in.close()
				del f_in
		del file_name
		
		assert file2Contents[orig_file] == file2Contents[new_file]
	
	def test_predefined_split_svmlight_file(self):
		###############################
		#<N.B.: For first run, did not clean up output files (which were copied to give the file copies to compare with in later test runs) and turned off comparison to file copies. These output files were manually inspected w.r.t. input files to check consistency with expectations.>
		###############################
		##############################
		print 'Running unittests for this project: ', project_name
		print 'Running this unittest: ', self._testMethodName
		##################################
		
		all_input_files_required_for_unittesting = []
		all_orig_output_files_to_be_compared_as_required_for_unittesting = []
		
		from ml_functions import predefined_split_svmlight_file
		
		train_ids_file = r'%s\TRAIN_onlyIDs-example-cp-from-test_5.txt' %  "\\".join(os.path.abspath(__file__).split('\\')[:-1])
		test_ids_file = r'%s\TEST_onlyIDs-example-cp-from-test_5.txt' % "\\".join(os.path.abspath(__file__).split('\\')[:-1])
		svmlight_allInstances_file = r'%s\svmlight_dataset-cp-from-test_5.txt' % "\\".join(os.path.abspath(__file__).split('\\')[:-1])
		
		subset2svmlightFile = predefined_split_svmlight_file(svmlight_file=svmlight_allInstances_file,train_IDs_file=train_ids_file,test_IDs_file=test_ids_file,output_label='unitTesting')
		
		svmlight_train_file = subset2svmlightFile['TRAIN']
		svmlight_test_file = subset2svmlightFile['TEST']
		del subset2svmlightFile
		
		return_var_contents_record  = r'%s\Names_of_svmlight_train_test_files.txt' % "\\".join(os.path.abspath(__file__).split('\\')[:-1])
		f_out = open(return_var_contents_record,'w')
		try:
			f_out.write('svmlight_train_file = %s\n'% svmlight_train_file.split(os.path.sep)[-1]) #08/01/2016 - fixed
			f_out.write('svmlight_test_file = %s\n'% svmlight_test_file.split(os.path.sep)[-1]) #08/01/2016 - fixed
		finally:
			f_out.close()
			del f_out
		
		
		#02/06/2013: commented out below for first trial runs and then, when output looked as expected, copied output and re-ran test with the following uncommented:
		#N.B. d.i. all of below prior to running - including compareOriginalAndNewFiles(...) - which needed to be corrected!!!
		all_input_files_required_for_unittesting += [svmlight_allInstances_file,train_ids_file,test_ids_file]
		
		
		for new_file in [svmlight_train_file,svmlight_test_file,return_var_contents_record]:
			file_ext = new_file.split('.')[-1]
			orig_file = re.sub('(\.%s$)' % file_ext,' - Copy.%s' % file_ext,new_file)
			all_orig_output_files_to_be_compared_as_required_for_unittesting.append(orig_file)
			self.compareOriginalAndNewFiles(orig_file,new_file)
		
		files_not_to_delete = all_input_files_required_for_unittesting+all_orig_output_files_to_be_compared_as_required_for_unittesting
		self.clean_up_if_all_checks_passed(specific_files_not_to_delete=files_not_to_delete)
