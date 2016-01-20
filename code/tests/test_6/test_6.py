#########################################################################################################
# test_6.py
# Implements unit tests for the genericQSARpyUtils project (see below).
#
# ########################################
# #test_6.py: Key documentation :Contents#
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

########################################################################
#<TO DO>: d.i.diff.w.r.t.test_5.py.after.test.run.fin.ok (all of below)#
########################################################################

import sys,re,os,glob
project_name = 'genericQSARpyUtils'
project_modules_to_test_dir =  "\\".join(os.path.abspath(__file__).split('\\')[:-3])
sys.path.append(project_modules_to_test_dir)
from ml_functions import crossValidate_svmlight_file
import unittest

class test_6(unittest.TestCase):
	
	def clean_up_if_all_checks_passed(self,specific_files_not_to_delete):
		
		all_files_to_delete = [file_name for file_name in glob.glob(r'%s\*' % "\\".join(os.path.abspath(__file__).split('\\')[:-1])) if not re.search('(.\py$)',file_name) and not file_name in specific_files_not_to_delete and not re.search('(Copy\.txt$)',file_name)]
		del file_name
		
		for FILE_TO_DELETE in all_files_to_delete:
			os.remove(FILE_TO_DELETE)
			assert not os.path.exists(FILE_TO_DELETE), " This still exists: \n %s" % FILE_TO_DELETE
			print 'Removed this temporary file: ', FILE_TO_DELETE
	
	def orderList(self,a_list):
		assert type([]) == type(a_list)
		a_list.sort()
		return a_list
	
	def ordered_file_lines(self,abs_file_name):
		f_in = open(abs_file_name)
		try:
			lines = [re.sub(r'\r|\n','',LINE) for LINE in f_in.readlines()]
			del LINE
			lines = self.orderList(lines)
			assert not 0 == len(lines)
		finally:
			f_in.close()
			del f_in
		return lines
	
	def apply_crossValidate_svmlight_file_function_to_contrived_input_using_different_settings(self,svmlight_file):
		#####################################################################################
		#N.B. When all regression y-values made different, the use of stratified=True led to error (scikit-learn version 0.13)!
		#=> "ValueError: The least populated class in y has only 1 member, which is too few. The minimum number of labels for any class cannot be less than 2."
		######################################################################################
		
		aTPF = {}
		
		#trial_stratified_mccv_keepEverything_partitionedFiles = crossValidate_svmlight_file(svmlight_file,output_dir= "\\".join(os.path.abspath(__file__).split('\\')[:-1]),mccv=True,folds=1,perc_test=0.2,repetitions=2,stratified=True,only_write_IDs=False)
		
		#aTPF['trial_stratified_mccv_keepEverything_partitionedFiles'] = trial_stratified_mccv_keepEverything_partitionedFiles
		
		trial_nonstratified_mccv_keepEverything_partitionedFiles = crossValidate_svmlight_file(svmlight_file,output_dir= "\\".join(os.path.abspath(__file__).split('\\')[:-1]),mccv=True,folds=1,perc_test=0.2,repetitions=2,stratified=False,only_write_IDs=False)
		
		aTPF['trial_nonstratified_mccv_keepEverything_partitionedFiles'] = trial_nonstratified_mccv_keepEverything_partitionedFiles
		
		#trial_stratified_mccv_keepIDs_partitionedFiles = crossValidate_svmlight_file(svmlight_file,output_dir= "\\".join(os.path.abspath(__file__).split('\\')[:-1]),mccv=True,folds=1,perc_test=0.2,repetitions=2,stratified=True,only_write_IDs=True)
		
		#aTPF['trial_stratified_mccv_keepIDs_partitionedFiles'] = trial_stratified_mccv_keepIDs_partitionedFiles
		
		trial_nonstratified_mccv_keepIDs_partitionedFiles = crossValidate_svmlight_file(svmlight_file,output_dir= "\\".join(os.path.abspath(__file__).split('\\')[:-1]),mccv=True,folds=1,perc_test=0.2,repetitions=2,stratified=False,only_write_IDs=True)
		
		aTPF['trial_nonstratified_mccv_keepIDs_partitionedFiles'] = trial_nonstratified_mccv_keepIDs_partitionedFiles
		
		#trial_stratified_kfoldcv_keepEverything_partitionedFiles = crossValidate_svmlight_file(svmlight_file,output_dir= "\\".join(os.path.abspath(__file__).split('\\')[:-1]),mccv=False,folds=5,perc_test=0.2,repetitions=2,stratified=True,only_write_IDs=False)
		
		#aTPF['trial_stratified_kfoldcv_keepEverything_partitionedFiles'] = trial_stratified_kfoldcv_keepEverything_partitionedFiles
		
		trial_nonstratified_kfoldcv_keepEverything_partitionedFiles = crossValidate_svmlight_file(svmlight_file,output_dir= "\\".join(os.path.abspath(__file__).split('\\')[:-1]),mccv=False,folds=5,perc_test=0.2,repetitions=2,stratified=False,only_write_IDs=False)
		
		aTPF['trial_nonstratified_kfoldcv_keepEverything_partitionedFiles'] = trial_nonstratified_kfoldcv_keepEverything_partitionedFiles
		
		#trial_stratified_kfoldcv_keepIDs_partitionedFiles = crossValidate_svmlight_file(svmlight_file,output_dir= "\\".join(os.path.abspath(__file__).split('\\')[:-1]),mccv=False,folds=5,perc_test=0.2,repetitions=2,stratified=True,only_write_IDs=True)
		
		#aTPF['trial_stratified_kfoldcv_keepIDs_partitionedFiles'] = trial_stratified_kfoldcv_keepIDs_partitionedFiles
		
		trial_nonstratified_kfoldcv_keepIDs_partitionedFiles = crossValidate_svmlight_file(svmlight_file,output_dir= "\\".join(os.path.abspath(__file__).split('\\')[:-1]),mccv=False,folds=5,perc_test=0.2,repetitions=2,stratified=False,only_write_IDs=True)
		
		aTPF['trial_nonstratified_kfoldcv_keepIDs_partitionedFiles'] = trial_nonstratified_kfoldcv_keepIDs_partitionedFiles
		
		return aTPF

	def check_crossValidate_svmlight_file_function_generates_train_test_files_whose_lines_are_consistent_with_the_original_dataset_file_and_do_not_overlap(self,svmlight_file,aTPF):
		
		
		f_in = open(svmlight_file)
		try:
			all_lines = [re.sub(r'\r|\n','',LINE) for LINE in f_in.readlines()]
			del LINE
			all_lines.sort()
			all_IDs = [LINE.split('#')[1] for LINE in all_lines]
			all_IDs.sort()
		finally:
			f_in.close()
			del f_in
		
		for scenario in aTPF:
			for rep in aTPF[scenario]:
				for FOLD in aTPF[scenario][rep]:
					print '-'*50
					print 'Considering train:test split for:'
					print 'scenario = ', scenario
					print 'rep = ', rep
					print 'FOLD = ' , FOLD
					print '-'*50
					
					linesDict = {}
					
					for subset in aTPF[scenario][rep][FOLD]:
						f_in = open(aTPF[scenario][rep][FOLD][subset])
						try:
							linesDict[subset] = [re.sub(r'\r|\n','',LINE) for LINE in f_in.readlines()]
							del LINE
						finally:
							f_in.close()
							del f_in
					
					assert 0 == len(set(linesDict['TRAIN']).intersection(set(linesDict['TEST'])))
					
					should_equal_all_lines_or_all_IDs = linesDict['TRAIN']+linesDict['TEST']
					should_equal_all_lines_or_all_IDs.sort()
					assert not 0 == len(should_equal_all_lines_or_all_IDs)
					
					if not re.search('(onlyIDs\.txt)',aTPF[scenario][rep][FOLD]['TRAIN']):
						assert all_lines == should_equal_all_lines_or_all_IDs, " Problem corresponds to this training file:\n %s \n all_lines = \n %s \n should_equal_all_lines_or_all_IDs = \n %s \n" % (aTPF[scenario][rep][FOLD]['TRAIN'],str(all_lines),str(should_equal_all_lines_or_all_IDs))
					else:
						assert all_IDs == should_equal_all_lines_or_all_IDs, " Problem corresponds to this training file:\n %s \n all_IDs = \n %s \n should_equal_all_lines_or_all_IDs = \n %s \n" % (aTPF[scenario][rep][FOLD]['TRAIN'],str(all_IDs),str(should_equal_all_lines_or_all_IDs))

	def test_crossValidate_svmlight_file_function_generates_train_test_files_whose_lines_are_consistent_with_the_original_dataset_file_and_do_not_overlap(self):
		##############################
		print 'Running unittests for this project: ', project_name
		print 'Running this unittest: ', self._testMethodName
		##################################
		
		
		
		
		svmlight_file = r'%s\checkcv_svmlight_train_file_for_regression.txt' % "\\".join(os.path.abspath(__file__).split('\\')[:-1])
		
		aTPF = self.apply_crossValidate_svmlight_file_function_to_contrived_input_using_different_settings(svmlight_file)
		
		########################################################
		#Function section which is specific to the current test:
		self.check_crossValidate_svmlight_file_function_generates_train_test_files_whose_lines_are_consistent_with_the_original_dataset_file_and_do_not_overlap(svmlight_file,aTPF)
		########################################################
		
		self.clean_up_if_all_checks_passed(specific_files_not_to_delete=[svmlight_file])
	
	# def test_crossValidate_svmlight_file_function_can_actually_give_different_results_if_switch_off_stratification(self):
		# ##############################
		# print 'Running unittests for this project: ', project_name
		# print 'Running this unittest: ', self._testMethodName
		# ##################################
		
		
		# #<DONE>: d.i.a.f.ok.r - including w.r.t. apply_crossValidate_svmlight_file_function_to_contrived_input_using_different_settings(self,svmlight_file) and def crossValidate_svmlight_file(...) where relevant to check that specifying criteria for stratified/non-stratified train/test file names.
		
		
		
		# svmlight_file = r'%s\checkcv_svmlight_train_file_for_regression.txt' % "\\".join(os.path.abspath(__file__).split('\\')[:-1])
		
		# aTPF = self.apply_crossValidate_svmlight_file_function_to_contrived_input_using_different_settings(svmlight_file)
		
		# ########################################################
		# #Function section which is specific to the current test:
		# all_strat_cv_scenarios = [key for key in aTPF.keys() if re.search('(_stratified_)',key)]
		# del key
		# assert 0.5*len(aTPF.keys()) == len(all_strat_cv_scenarios)
		
		# for strat_cv_scenario in all_strat_cv_scenarios:
			# for rep in aTPF[strat_cv_scenario]:
				# for FOLD in aTPF[strat_cv_scenario][rep]:
					# for subset in ['TRAIN','TEST']:
						# corresponding_non_strat_cv_scenario = re.sub('(_stratified_)','_nonstratified_',strat_cv_scenario)
						# assert not self.ordered_file_lines(aTPF[strat_cv_scenario][rep][FOLD][subset]) == self.ordered_file_lines(aTPF[corresponding_non_strat_cv_scenario][rep][FOLD][subset])
		
		# ########################################################
		
		# self.clean_up_if_all_checks_passed(specific_files_not_to_delete=[svmlight_file])
	
	def test_crossValidate_svmlight_file_function_can_actually_give_different_partitions_for_different_reps(self):
		##############################
		print 'Running unittests for this project: ', project_name
		print 'Running this unittest: ', self._testMethodName
		##################################
		
		
		
		
		
		
		svmlight_file = r'%s\checkcv_svmlight_train_file_for_regression.txt' % "\\".join(os.path.abspath(__file__).split('\\')[:-1])
		
		aTPF = self.apply_crossValidate_svmlight_file_function_to_contrived_input_using_different_settings(svmlight_file)
		
		########################################################
		#Function section which is specific to the current test:
		for scenario in aTPF:
			for rep in aTPF[scenario]:
				for other_rep in aTPF[scenario].keys()[aTPF[scenario].keys().index(rep)+1:]:
					assert not rep == other_rep
					for FOLD in aTPF[scenario][rep]:
						for same_or_different_fold in aTPF[scenario][rep]:
							for subset in ['TRAIN','TEST']:
								assert not self.ordered_file_lines(aTPF[scenario][rep][FOLD][subset]) == self.ordered_file_lines(aTPF[scenario][other_rep][same_or_different_fold][subset])
		
		########################################################
		
		self.clean_up_if_all_checks_passed(specific_files_not_to_delete=[svmlight_file])
	
	def test_reproducibility_of_crossValidate_svmlight_file_function(self):
		##############################
		print 'Running unittests for this project: ', project_name
		print 'Running this unittest: ', self._testMethodName
		##################################
		
		
		svmlight_file = r'%s\checkcv_svmlight_train_file_for_regression.txt' % "\\".join(os.path.abspath(__file__).split('\\')[:-1])
		
		aTPF = self.apply_crossValidate_svmlight_file_function_to_contrived_input_using_different_settings(svmlight_file)
		
		########################################################
		#Function section which is specific to the current test:
		for scenario in aTPF:
			for rep in aTPF[scenario]:
				for FOLD in aTPF[scenario][rep]:
					for subset in ['TRAIN','TEST']:
						assert self.ordered_file_lines(aTPF[scenario][rep][FOLD][subset]) == self.ordered_file_lines(re.sub('(\.txt$)',' - Copy.txt',aTPF[scenario][rep][FOLD][subset])), "%s and %s are different???" % (aTPF[scenario][rep][FOLD][subset],re.sub('(\.txt$)',' - Copy.txt',aTPF[scenario][rep][FOLD][subset]))
		
		########################################################
		
		self.clean_up_if_all_checks_passed(specific_files_not_to_delete=[svmlight_file])
	
	def test_crossValidate_svmlight_file_function_only_IDs_files_consistent(self):
		##############################
		print 'Running unittests for this project: ', project_name
		print 'Running this unittest: ', self._testMethodName
		##################################
		
		
		svmlight_file = r'%s\checkcv_svmlight_train_file_for_regression.txt' % "\\".join(os.path.abspath(__file__).split('\\')[:-1])
		
		aTPF = self.apply_crossValidate_svmlight_file_function_to_contrived_input_using_different_settings(svmlight_file)
		
		########################################################
		#Function section which is specific to the current test:
		for scenario in aTPF:
			for rep in aTPF[scenario]:
				for FOLD in aTPF[scenario][rep]:
					for subset in ['TRAIN','TEST']:
						if re.search('(_onlyIDs\.txt$)',aTPF[scenario][rep][FOLD][subset]):
							assert self.ordered_file_lines(aTPF[scenario][rep][FOLD][subset]) == self.orderList([LINE.split('#')[1] for LINE in self.ordered_file_lines(re.sub('(_onlyIDs\.txt$)','.txt',aTPF[scenario][rep][FOLD][subset]))]) #, " This pair of files is inconsistent: \n %s \n vs. \n %s \n first file list = \n %s \n second file list = %s \n" % (
							del LINE
		
		########################################################
		
		self.clean_up_if_all_checks_passed(specific_files_not_to_delete=[svmlight_file])
