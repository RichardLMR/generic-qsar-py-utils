#########################################################################################################
# test_7.py
# Implements unit tests for the genericQSARpyUtils project (see below).
#
# ########################################
# #test_7.py: Key documentation :Contents#
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
project_modules_to_test_dir = "\\".join(os.path.abspath(__file__).split('\\')[:-3])
sys.path.append(project_modules_to_test_dir)


import unittest
class test_7(unittest.TestCase):
	#17/03/13: Adapted from test_4.
	
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
	
	def test_univariate_training_set_based_feature_selection_for_raw_fp_representations_of_training_and_test_set_for_a_REGRESSION_dataset(self):
		###############################
		#17/03/13:
		#<N.B.: Using exactly the same input files as per test_4.>
		#<N.B.: For first run, did not clean up output files (which were copied to give the file copies to compare with in later test runs) and turned off comparison to file copies.>
		###############################
		##############################
		print 'Running unittests for this project: ', project_name
		print 'Running this unittest: ', self._testMethodName
		##################################
		
		#Note to self: BELOW taken verbatim from ..\trial_runs\...\trial_run_fs_2.py
		#from descriptor_utils import descriptorsGenerator,descriptorsFilesProcessor #Note to self: replaced with the next line.
		from ml_input_utils import descriptorsFilesProcessor
		from ml_functions import filter_features_for_svmlight_format_files,f_regression
		
		id2TrainYValue = {'mA':1.8,'mB':1.8,'mC':0.1,'mD':1.8,'mE':0.1,'mG':1.8,'mF':0.1,'mH':1.8} #17/03/13: trying to make sure f2 has a perfect correlation with y-values in training set bu no such correlation [due to constant y-values] in the test set (see next line). <<DONE>:D.I.P.T.R>
		id2TestYValue = {'mX':1.0,'mY':1.0,'mZ':1.0}
		#Note to self: as ever, following file names need to be adjusted to make sure files in the directory of this test code Python file are parsed.
		train_fp_file = r'%s\contrived_fp_train_file.txt' % "\\".join(os.path.abspath(__file__).split('\\')[:-1])
		test_fp_file = r'%s\contrived_fp_test_file.txt' % "\\".join(os.path.abspath(__file__).split('\\')[:-1])
		all_feats_svmlight_train_file = r'%s\contrived_svmlight_train_file.txt' % "\\".join(os.path.abspath(__file__).split('\\')[:-1])
		all_feats_svmlight_test_file = r'%s\contrived_svmlight_test_file.txt' % "\\".join(os.path.abspath(__file__).split('\\')[:-1])
		
		
		print 'Preparing original files (pre-feature selection) in svmlight format.'
		
		
		our_descriptorsFilesProcessor = descriptorsFilesProcessor()
		
		record_of_all_feat2IndexFiles = [None]
		
		for TRAIN_OR_TEST_LABEL in ['Train','Test']:
			if 'Train' == TRAIN_OR_TEST_LABEL:
				id2class = id2TrainYValue
				fp_file = train_fp_file
				all_feats_svmlight_file =  all_feats_svmlight_train_file
			else:
				assert 'Test' == TRAIN_OR_TEST_LABEL
				id2class = id2TestYValue
				fp_file = test_fp_file
				all_feats_svmlight_file = all_feats_svmlight_test_file
			
			record_of_all_feat2IndexFiles = our_descriptorsFilesProcessor.write_svmlight_format_modellingFile_from_multiple_descriptors_files(list_of_descriptors_files=[fp_file],corresponding_list_of_whether_descriptors_file_is_actually_a_raw_fp_file=[True],corresponding_list_of_whether_descriptors_file_is_actually_a_jCompoundMapperStringFeatures_file=[False],descriptors_file_name=all_feats_svmlight_file,id2responseVariable=id2class,corresponding_list_of_unique_features_files=record_of_all_feat2IndexFiles)
		
		del our_descriptorsFilesProcessor
		
		print 'PREPARED original files (pre-feature selection) in svmlight format.'
		
		######
		#<10/10/12::16:45: N.B.: INSPECTION OF ABOVE OUTPUT =>  our_descriptorsFilesProcessor.write_svmlight_format_modellingFile_from_multiple_descriptors_files(...) WORKS!>
		######
		
		filter_features_for_svmlight_format_files(svmlight_format_train_file=all_feats_svmlight_train_file,svmlight_format_test_file=all_feats_svmlight_test_file,univariate_scoring_function=f_regression,number_of_features_to_retain=2)
		
		#Note to self: ABOVE taken verbatim from ..\trial_runs\...\trial_run_fs_2.py
		
		filtered_feats_svmlight_train_file = r'%s\contrived_svmlight_train_file_fs_f_regression_top_2.txt' % "\\".join(os.path.abspath(__file__).split('\\')[:-1]) 
		filtered_feats_svmlight_test_file = r'%s\contrived_svmlight_test_file_fs_f_regression_top_2.txt' % "\\".join(os.path.abspath(__file__).split('\\')[:-1]) 
		
		
		#23/03/13: commented out below for first trial runs and then, when output looked as expected<ok>, copied output and re-ran test with the following uncommented:
		all_input_files_required_for_unittesting = [train_fp_file,test_fp_file]
		
		all_orig_output_files_to_be_compared_as_required_for_unittesting = []
		for new_file in [all_feats_svmlight_train_file,all_feats_svmlight_test_file,filtered_feats_svmlight_train_file,filtered_feats_svmlight_test_file]:
			file_ext = new_file.split('.')[-1]
			orig_file = re.sub('(\.%s$)' % file_ext,' - Copy.%s' % file_ext,new_file)
			all_orig_output_files_to_be_compared_as_required_for_unittesting.append(orig_file)
			self.compareOriginalAndNewFiles(orig_file,new_file)
		
		files_not_to_delete = all_input_files_required_for_unittesting+all_orig_output_files_to_be_compared_as_required_for_unittesting
		self.clean_up_if_all_checks_passed(specific_files_not_to_delete=files_not_to_delete)
