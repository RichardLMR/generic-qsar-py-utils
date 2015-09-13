#########################################################################################################
# test_8.py
# Implements unit tests for the genericQSARpyUtils project (see below).
#
# ########################################
# #test_8.py: Key documentation :Contents#
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
import sys,re,os,glob
project_name = 'genericQSARpyUtils'
project_modules_to_test_dir = "\\".join(os.path.abspath(__file__).split('\\')[:-3])
sys.path.append(project_modules_to_test_dir)


import unittest
class test_8(unittest.TestCase):
	
	################################################################
	#========================================================
	#Description of testing carried out by this test class.
	#========================================================
	#This test is designed to check the *no-default* behaviour of ml_functions.py: filter_features_for_svmlight_format_files(...),
	#enabled when ensure_test_set_consistency=True.
	#By default, if there are test set features which are not present in those selected based upon the training set, the set of training set and test set features are inconsistent. However, if the non-default option ensure_test_set_consistency=True is used, these missing training set features are added to the svmlight-format output file for the test set with a value of zero.
	#Notes:
	#The default behaviour is retained 
	#(1) to avoid the *possibility*(?) of breaking older unit tests for which selected features were also not present in the test set.
	#(2) because this issue only becomes a problem when trying to create consistent training and test set csv files using ml_input_utils.py:descriptorsFilesProcessor.convert_svmlight_to_csv(...) - whereas svmlight format can handle missing features as they are assumed to have a value of zero as would be the case when, as here, we are considering fingerprint features represented as present (value=1) or absent (value=0).
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
	
	def test_non_default_FS_which_adds_missing_validation_set_features(self):
		###############################
		#<N.B.: For first run, did not clean up output files (which were copied to give the file copies to compare with in later test runs) and turned off comparison to file copies. These output files were manually inspected w.r.t. input files to check consistency with expectations.>
		###############################
		##############################
		print 'Running unittests for this project: ', project_name
		print 'Running this unittest: ', self._testMethodName
		##################################
		
		all_input_files_required_for_unittesting = []
		all_orig_output_files_to_be_compared_as_required_for_unittesting = []
		
		from ml_functions import filter_features_for_svmlight_format_files,f_regression,chi2
		
		NUMBER_OF_KEPT_FEATURES = 2
		
		for NOT_DEFAULT in [True,False]:
			for REG_OR_CLASS in ['REG','CLASS']:
				if 'REG' == REG_OR_CLASS:
					FS_FUNCTION = f_regression
					fs_method_label = 'f_regression'
				else:
					assert 'CLASS' == REG_OR_CLASS
					FS_FUNCTION = chi2
					fs_method_label = 'chi2'
				
				all_feats_svmlight_train_file = r'%s\svmlight_%s_train_file.txt' % ("\\".join(os.path.abspath(__file__).split('\\')[:-1]),REG_OR_CLASS)
				all_feats_svmlight_test_file = r'%s\svmlight_%s_test_file.txt' % ("\\".join(os.path.abspath(__file__).split('\\')[:-1]),REG_OR_CLASS)
				
				
				filter_features_for_svmlight_format_files(svmlight_format_train_file=all_feats_svmlight_train_file,svmlight_format_test_file=all_feats_svmlight_test_file,univariate_scoring_function=FS_FUNCTION,number_of_features_to_retain=NUMBER_OF_KEPT_FEATURES,ensure_test_set_consistency=NOT_DEFAULT)
				
				
				#Names must be consistent with train/test file and options passed to filter_features_for_svmlight_format_files(...):
				
				if NOT_DEFAULT:
					DEFAULT_OR_NOT_LABEL = '_nonDefault_fs'
				else:
					DEFAULT_OR_NOT_LABEL = '_fs'
				
				filtered_feats_svmlight_train_file = r'%s\svmlight_%s_train_file%s_%s_top_%d.txt' % ("\\".join(os.path.abspath(__file__).split('\\')[:-1]),REG_OR_CLASS,DEFAULT_OR_NOT_LABEL,fs_method_label,NUMBER_OF_KEPT_FEATURES) 
				filtered_feats_svmlight_test_file = r'%s\svmlight_%s_test_file%s_%s_top_%d.txt' % ("\\".join(os.path.abspath(__file__).split('\\')[:-1]),REG_OR_CLASS,DEFAULT_OR_NOT_LABEL,fs_method_label,NUMBER_OF_KEPT_FEATURES)
				
				
				#02/05/2013: commented out below for first trial runs and then, when output looked as expected<TO DO:PARTIALLY DONE:TEST SETS LOOK AS EXPECTED WHEN LABELLED 'nonDefault' OR NOT>, copied output and re-ran test with the following uncommented:
				all_input_files_required_for_unittesting += [all_feats_svmlight_train_file,all_feats_svmlight_test_file]
				
				
				for new_file in [filtered_feats_svmlight_train_file,filtered_feats_svmlight_test_file]:
					file_ext = new_file.split('.')[-1]
					orig_file = re.sub('(\.%s$)' % file_ext,' - Copy.%s' % file_ext,new_file)
					all_orig_output_files_to_be_compared_as_required_for_unittesting.append(orig_file)
					self.compareOriginalAndNewFiles(orig_file,new_file)
				
		files_not_to_delete = all_input_files_required_for_unittesting+all_orig_output_files_to_be_compared_as_required_for_unittesting
		self.clean_up_if_all_checks_passed(specific_files_not_to_delete=files_not_to_delete)
