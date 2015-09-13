#########################################################################################################
# test_1.py
# Implements unit tests for the genericQSARpyUtils project (see below).

# ########################################
# #test_1.py: Key documentation :Contents#
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
project_modules_to_test_dir = "\\".join(os.path.abspath(__file__).split('\\')[:-3]) #16/01/13:14:42: No other changes to *code*, but "-2" replace here with "-3" as now each test is placed in its own subdirectory!
sys.path.append(project_modules_to_test_dir)


import unittest
class test_1(unittest.TestCase):
	
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
		
		try:
			assert file2Contents[orig_file] == file2Contents[new_file], " These files do not match: \n %s \n %s \n" % (orig_file,new_file)
		except AssertionError,msg:
			#################################################################################################
			#15/01/13: It appears that jCompoundMapper fingerprint files (original vs. generated when run this test) may not maintain the *order* of the features associated with a given molecule id -BUT the features remain the same!
			#The following lines are designed to take account of such "not genuine file content inconsistencies".
			#Unfortunately, the ordering of the features in each line determines the exact number of the index assigned to each feature - which determines the exact contents of the final Machine Learning input file!!!! => 1. Modified the contents of ml_input_utils.py to make sure that the features associated with an ID are canonically ordered prior to assigning feature indices. 2. Manually deleted old output from last run of the current test. 3.Re-ran the current test. 4. Copied the final Machine Learning input file to give the reference version to which all further runs of the current test should make comparisons to!
			#################################################################################################
			print '='*50
			print msg
			del msg
			assert new_file.split("\\")[-1] == 'trial_smiles_input_PP_STAND_pybel_IDs_-aDAYLIGHT_INVARIANT_RING-cECFP-d2-ffSTRING_PATTERNS.txt'
			
			orig_id2features = {}
			for LINE in file2Contents[orig_file].split('<EOL>'):
				orig_id2features[LINE.split('\t')[0]] = LINE.split('\t')[1:]
				orig_id2features[LINE.split('\t')[0]].sort()
			del LINE
			
			new_id2features = {}
			for LINE in file2Contents[new_file].split('<EOL>'):
				new_id2features[LINE.split('\t')[0]] = LINE.split('\t')[1:]
				new_id2features[LINE.split('\t')[0]].sort()
			del LINE
			
			assert new_id2features.keys() == orig_id2features.keys()
			
			for ID in orig_id2features:
				assert orig_id2features[ID] == new_id2features[ID] , " Different features list for ID = %s!" % ID#15/01/13: <*TO DO*- ???: FOLLOW THIS UP> I am a bit concerned that I may have once observed this to fail randomly (i.e. ran run_tests.py mutiple times without test failures ... then this failed sometimes under the same setup....) BUT I may be mistaken?
			del ID
			print 'But all ID:<list of features> pairings do match once features are consistently ordered!'
			print '='*50
		
	
	
	def generate_svmlightMachineLearningInput_BasedOn_NumericDescriptors_PLUS_jCompoundMapper_And_Murcko_FingerprintBitVector(self,trial_smiles_input,trial_corresponding_numeric_descriptors_csv,trial_corresponding_class_labels_csv,svmlight_ml_input_file_name):
		
		def get_class_labels():
			f_in = open(trial_corresponding_class_labels_csv)
			try:
				lines = [re.sub(r'\r|\n','',LINE) for LINE in f_in.readlines()[1:]]
				del LINE
			finally:
				f_in.close()
				del f_in
			id2class = dict([(LINE.split(',')[0],int(LINE.split(',')[1])) for LINE in lines])
			
			return id2class
		
		#####################
		from ml_input_utils import descriptorsGenerator
		our_descriptorsGenerator = descriptorsGenerator(molecules_input_file=trial_smiles_input,output_dir=r'%s' % "\\".join(os.path.abspath(__file__).split('\\')[:-1]))
		#####################
		
		#Calculate descriptors:
		
		jCompoundMapper_fp_file = our_descriptorsGenerator.write_jCompoundMapper_fp_file(atom_type='DAYLIGHT_INVARIANT_RING',fp_type='ECFP',distance_cutoff_or_search_depth=2,output_format='STRING_PATTERNS')
		
		scaffolds_fp_file = our_descriptorsGenerator.write_scaffolds_fp_file()
		
		################################
		del our_descriptorsGenerator
		del descriptorsGenerator
		################################
		
		#############################
		from ml_input_utils import descriptorsFilesProcessor
		our_descriptorsFilesProcessor = descriptorsFilesProcessor()
		##############################
		
		#Match descriptor values to molecule IDs:
		
		combined_molId2DescId2DescValue = our_descriptorsFilesProcessor.generate_molId2DescId2DescValue_from_multiple_descriptors_files(list_of_descriptors_files=[jCompoundMapper_fp_file,scaffolds_fp_file,trial_corresponding_numeric_descriptors_csv],corresponding_list_of_whether_descriptors_file_is_actually_a_raw_fp_file=[True,True,False],corresponding_list_of_whether_descriptors_file_is_actually_a_jCompoundMapperStringFeatures_file=[True,False,False])
		
		#Match class labels to molecule IDs:
		#Class labels MUST be converted to a numeric format!
		
		id2class = get_class_labels()
		
		#Write modelling input file in two different formats:
		
		our_descriptorsFilesProcessor.write_svmlight_format_modellingFile_for_generic_descriptors(molId2DescId2DescValue=combined_molId2DescId2DescValue,descriptors_file_name=svmlight_ml_input_file_name,id2responseVariable=id2class)
		
		return jCompoundMapper_fp_file,scaffolds_fp_file
	
	def test_generate_svmlightMachineLearningInput_BasedOn_NumericDescriptors_PLUS_FingerprintBitVector(self):
		##############################
		print 'Running unittests for this project: ', project_name
		print 'Running this unittest: ', self._testMethodName
		##################################
		#<DONE>:<PROPOSED FINAL MANUAL CHECKING OF THE OUTPUT OF THE WORKFLOW RAN BY THIS TEST>:After the last *code* [i.e. comments may be updated afterwards] modification (both of the functionality in ml_input_utils.py employed in this test and the current test itself)  followed by running this test => test runs to completion without failures, MANUALLY CHECKED that svmlight_trial_model_input - Copy.txt (i.e. the reference version of svmlight_ml_input_file_name to which the temporary version upon running this test is asserted to match) is consistent with (a) all files in all_input_files_required_for_unittesting, (b) all reference versions of the intermediate output files - i.e. raw fingerprint files - referred to in all_orig_output_files_to_be_compared_as_required_for_unittesting, based on expected output (informed via some inspection of ml_input_utils.py). <<OUTCOME OF THIS MANUAL CHECKING>: ALL svmlight file line number: class label / INDEX+VAMP LUMO DESCRIPTOR VALUE/ INDEX+scaffold FP FEATURE DESCRIPTOR VALUE  correspondence OK. Ditto maximum descriptor index. <N.B.>:<In constrast to the proposal>It was necessary to determine whether svmlight file line number: INDEX+jCompoundMapper FP FEATURE DESCRIPTOR VALUE correspondence was OK based on cross-referencing fingerprint file (...Copy.txt version) w.r.t. intermediate trial_smiles_input_PP_STAND_pybel_IDs_-aDAYLIGHT_INVARIANT_RING-cECFP-d2-ffSTRING_PATTERNS_fpFeat2InitialIndex.csv generated when running this test [I.E. COMMENTED OUT self.clean_up_if_all_checks_passed(specific_files_not_to_delete=files_not_to_delete), THEN UNCOMMENTED AFTER MANUAL CHECKING AND RAN THIS TEST AGAIN...]  - as order of features could not be reproduced within Excel. After this cross-referencing => correspondence = OK. Also checked svmlight_trial_model_input_recordOfIDsOrder.txt contents against expected molID: svmlight file line correspondence based upon considering ml_input_utils.py code => OK>
		###################################
		##############################
		
		
		trial_smiles_input = r'%s\trial_smiles_input_PP_STAND.smi' % "\\".join(os.path.abspath(__file__).split('\\')[:-1])
		trial_corresponding_numeric_descriptors_csv = r'%s\trial_input_PP_STAND_PP_vamp_descriptors_file.csv' % "\\".join(os.path.abspath(__file__).split('\\')[:-1])
		trial_corresponding_class_labels_csv = r'%s\trial_class_labels.csv' % "\\".join(os.path.abspath(__file__).split('\\')[:-1])
		
		all_input_files_required_for_unittesting = [trial_smiles_input,trial_corresponding_numeric_descriptors_csv,trial_corresponding_class_labels_csv]
		
		svmlight_ml_input_file_name=r'%s\svmlight_trial_model_input.txt' % "\\".join(os.path.abspath(__file__).split('\\')[:-1])
		
		
		
		jCompoundMapper_fp_file,scaffolds_fp_file = self.generate_svmlightMachineLearningInput_BasedOn_NumericDescriptors_PLUS_jCompoundMapper_And_Murcko_FingerprintBitVector(trial_smiles_input,trial_corresponding_numeric_descriptors_csv,trial_corresponding_class_labels_csv,svmlight_ml_input_file_name)
		
		
		
		all_orig_output_files_to_be_compared_as_required_for_unittesting = []
		for new_file in [jCompoundMapper_fp_file,scaffolds_fp_file,svmlight_ml_input_file_name]:
			file_ext = new_file.split('.')[-1]
			orig_file = re.sub('(\.%s$)' % file_ext,' - Copy.%s' % file_ext,new_file)
			all_orig_output_files_to_be_compared_as_required_for_unittesting.append(orig_file)
			self.compareOriginalAndNewFiles(orig_file,new_file)
		
		files_not_to_delete = all_input_files_required_for_unittesting+all_orig_output_files_to_be_compared_as_required_for_unittesting
		self.clean_up_if_all_checks_passed(specific_files_not_to_delete=files_not_to_delete)
