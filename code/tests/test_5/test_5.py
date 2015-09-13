#########################################################################################################
# test_5.py
# Implements unit tests for the genericQSARpyUtils project (see below).

# ########################################
# #test_5.py: Key documentation :Contents#
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
# #To this end, two Python modules are currently (as of 17/01/2013) provided. 
# #(1) ml_input_utils.py 
# #Defines two classes:
# #(i)descriptorsGenerator: This contains methods which can be used (as of 17/01/2013) to interconvert between molecular file formats (e.g. SDF, SMILES,...),
# write the molecule ID to an SDF field, as well as calculate fingerprints presenting raw text codes for substructural features (e.g. extended connectivity fingerprints using jCompoundMapper or scaffold fragment fingerprints).
# #(ii)descriptorsFilesProcessor: This contains methods which can be used (as of 17/01/2013) to convert raw fingerprint files 
# #(i.e. files with a .txt extension in which each line corresponds to a molecule and has the following form:
# #molId<TAB>FeatureB<TAB>FeatureC<TAB>FeatureA<TAB>FeatureX.... where FeatureB etc. is raw text string) into
# #Machine Learning modelling input files (in either svmlight or csv format) where the features are represented using
# #a bit-string encoding. (Here, a bi-string encoding means a descriptor corresponding to each - of a specifed set - of
# #features found in the dataset, with the descriptor value being 1 or 0 if the feature was present or absent in a given molecule.)
# #The methods in this class also allow for additional descriptors (e.g. a set of ClogP values) to be added to the modelling
# #input files.
# #(2) ml_functions.py
# #Defines a set of functions which can be used (as of 17/01/2013) to carry out univariate feature selection
# #and MonteCarlo cross-validation (which, for a single repetition, corresponds to a single train:test partition)
# #for Machine Learning model input files in svmlight format.

# ###########################
# #2. IMPORTANT LEGAL ISSUES#
# ###########################
# Copyright Syngenta Limited 2013
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
class test_5(unittest.TestCase):
	
	def clean_up_if_all_checks_passed(self,specific_files_not_to_delete):
		all_files_to_delete = [file_name for file_name in glob.glob(r'%s\*' % "\\".join(os.path.abspath(__file__).split('\\')[:-1])) if not re.search('(.\py$)',file_name) and not file_name in specific_files_not_to_delete]
		
		for FILE_TO_DELETE in all_files_to_delete:
			os.remove(FILE_TO_DELETE)
			assert not os.path.exists(FILE_TO_DELETE), " This still exists: \n %s" % FILE_TO_DELETE
			print 'Removed this temporary file: ', FILE_TO_DELETE
	
	def apply_mccv_svmlight_function_to_contrived_input_using_different_settings(self,svmlight_file):
		#<DONE>: d.i.p.t.r (including w.r.t. argument specification and reporting sections at start of def mccv_svmlight_file(...):)
		from ml_functions import mccv_svmlight_file
		
		different_trial_settings = [[1,True],[2,True],[1,False],[2,False]]
		
		all_partitionedFiles = []
		
		for trial_number in range(0,len(different_trial_settings)):
			
			
			
			print '#'*50
			
			print 'Trial number: ', trial_number
			
			partitionedFiles = mccv_svmlight_file(svmlight_file,output_dir="\\".join(os.path.abspath(__file__).split('\\')[:-1]),perc_test=float(1.0/3),repetitions=different_trial_settings[trial_number][0],stratified=different_trial_settings[trial_number][1])
			
			
			
			print '\n'+'*'*50
			print 'Names of created files: '
			for rep in range(1,1+different_trial_settings[trial_number][0]):
				for subset in ['TRAIN','TEST']:
					print '-'*50
					print 'Repetition: ', rep
					print 'Subset: ', subset
					print 'File name: ', partitionedFiles[rep][subset]
					print '-'*50
			print '\n'+'*'*50
			print '#'*50
			
			all_partitionedFiles.append(partitionedFiles)
		
		assert not 0 == len(all_partitionedFiles)
		assert len(different_trial_settings) == len(all_partitionedFiles)
		
		return all_partitionedFiles
	
	def check_mccv_svmlight_file_function_always_generates_train_test_files_whose_lines_are_consistent_with_the_original_dataset_file_and_do_not_overlap(self,svmlight_file,all_partitionedFiles):
		#<DONE>: d.i.p.t.r - including w.r.t. def apply_mccv_svmlight_function_to_contrived_input_using_different_settings(self,svmlight_file) and [where relevant to looping over setOfTrainTestFiles syntax], w.r.t. def mccv_svmlight_file(svmlight_file,output_dir=os.getcwd(),perc_test=0.2,repetitions=1,stratified=True,rng_seed=0):
		count = 0
		for setOfTrainTestFiles in all_partitionedFiles:
			count +=1
			for split_number in setOfTrainTestFiles:
				print '#'*50
				print 'setOfTrainTestFiles number %d' % count
				print 'split_number = %d' % split_number
				print 'Checking (1) that this training set file:' 
				print setOfTrainTestFiles[split_number]['TRAIN']
				print 'does not overlap with this test set file:'
				print setOfTrainTestFiles[split_number]['TEST']
				
				subset2Lines = {}
				for SUBSET in ['TRAIN','TEST']:
					f_in = open(setOfTrainTestFiles[split_number][SUBSET])
					try:
						subset2Lines[SUBSET] = [re.sub(r'\r|\n','',LINE) for LINE in f_in.readlines()]
						del LINE
					finally:
						f_in.close()
						del f_in
				assert 0 == len(set(subset2Lines['TRAIN']).intersection(set(subset2Lines['TEST'])))
				
				print 'Checking (2) that total dataset lines matched the sum of the training and test set lines.'
				
				train_plus_test_lines = subset2Lines['TRAIN']+subset2Lines['TEST']
				del subset2Lines
				train_plus_test_lines.sort()
				
				f_in = open(svmlight_file)
				try:
					all_lines_read_from_original_file = [re.sub(r'\r|\n','',LINE) for LINE in f_in.readlines()]
					del LINE
				finally:
					f_in.close()
					del f_in
				
				all_lines_read_from_original_file.sort()
				
				assert all_lines_read_from_original_file == train_plus_test_lines
				del train_plus_test_lines
				print '#'*50
	
	def test_mccv_svmlight_file_function_always_generates_train_test_files_whose_lines_are_consistent_with_the_original_dataset_file_and_do_not_overlap(self):
		##############################
		print 'Running unittests for this project: ', project_name
		print 'Running this unittest: ', self._testMethodName
		##################################
		
		
		#<DONE>: d.i.p.t.r - including w.r.t. apply_mccv_svmlight_function_to_contrived_input_using_different_settings(self,svmlight_file) and check_mccv_svmlight_file_function_always_generates_train_test_files_whose_lines_are_consistent_with_the_original_dataset_file_and_do_not_overlap(...)
		
		svmlight_file = r'%s\contrived_svmlight_train_file.txt' % "\\".join(os.path.abspath(__file__).split('\\')[:-1])
		
		all_partitionedFiles = self.apply_mccv_svmlight_function_to_contrived_input_using_different_settings(svmlight_file)
		
		########################################################
		#Function section which is specific to the current test:
		self.check_mccv_svmlight_file_function_always_generates_train_test_files_whose_lines_are_consistent_with_the_original_dataset_file_and_do_not_overlap(svmlight_file,all_partitionedFiles)
		########################################################
		
		self.clean_up_if_all_checks_passed(specific_files_not_to_delete=[svmlight_file])
	
	def test_mccv_svmlight_file_function_can_actually_give_different_results_if_switch_off_stratification(self):
		##############################
		print 'Running unittests for this project: ', project_name
		print 'Running this unittest: ', self._testMethodName
		##################################
		
		
		#<DONE>: d.i.p.t.r - including w.r.t. apply_mccv_svmlight_function_to_contrived_input_using_different_settings(self,svmlight_file) and def mccv_svmlight_file(...) where relevant to check that specificying criteria for stratified/non-stratified train/test file names.
		
		
		
		svmlight_file = r'%s\contrived_svmlight_train_file.txt' % "\\".join(os.path.abspath(__file__).split('\\')[:-1])
		
		all_partitionedFiles = self.apply_mccv_svmlight_function_to_contrived_input_using_different_settings(svmlight_file)
		
		########################################################
		#Function section which is specific to the current test:
		#different_trial_settings = [[1,True],[2,True],[1,False],[2,False]] #copied from def apply_mccv_svmlight_function_to_contrived_input_using_different_settings(self,svmlight_file)
		all_stratifiedPartitionFiles = [] 
		for TRIAL_NUMBER in range(0,len(all_partitionedFiles)):
			for rep in all_partitionedFiles[TRIAL_NUMBER]:
				for subset in all_partitionedFiles[TRIAL_NUMBER][rep]:
					assert re.search('(R%d%s)' % (rep,subset),all_partitionedFiles[TRIAL_NUMBER][rep][subset])
					if TRIAL_NUMBER in [0,1]:
						assert re.search('(STrue)',all_partitionedFiles[TRIAL_NUMBER][rep][subset])
						all_stratifiedPartitionFiles.append(all_partitionedFiles[TRIAL_NUMBER][rep][subset])
					else:
						assert re.search('(SFalse)',all_partitionedFiles[TRIAL_NUMBER][rep][subset])
		del TRIAL_NUMBER
		del rep
		del subset
		
		for stratifiedPartitionFILE in all_stratifiedPartitionFiles:
			corresponding_nonstratifiedPartitionFILE = re.sub('(STrue)','SFalse',stratifiedPartitionFILE)
			
			print '='*50
			print 'Comparing:'
			print stratifiedPartitionFILE
			print 'and:'
			print corresponding_nonstratifiedPartitionFILE
			
			stratifiedPartitionFILE_lines_THEN_corresponding_nonstratifiedPartitionFILE_lines = []
			
			for FILE in [stratifiedPartitionFILE,corresponding_nonstratifiedPartitionFILE]:
				
				f_in = open(FILE)
				try:
					lines = [re.sub(r'\r|\n','',LINE) for LINE in f_in.readlines()]
					del LINE
				finally:
					f_in.close()
				lines.sort()
				
				stratifiedPartitionFILE_lines_THEN_corresponding_nonstratifiedPartitionFILE_lines.append(lines)
				del lines
			
			assert not stratifiedPartitionFILE_lines_THEN_corresponding_nonstratifiedPartitionFILE_lines[0] == stratifiedPartitionFILE_lines_THEN_corresponding_nonstratifiedPartitionFILE_lines[1]
			print '='*50
		
		########################################################
		
		self.clean_up_if_all_checks_passed(specific_files_not_to_delete=[svmlight_file])
