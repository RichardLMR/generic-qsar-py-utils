#########################################################################################################
# test_2.py
# Implements unit tests for the genericQSARpyUtils project (see below).

# ########################################
# #test_2.py: Key documentation :Contents#
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
class test_2(unittest.TestCase):
	
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
	
	def test_only_retain_entries_in_descriptors_files_corresponding_to_common_IDs(self):
		##############################
		print 'Running unittests for this project: ', project_name
		print 'Running this unittest: ', self._testMethodName
		##################################
		
		#Note to self: BELOW copied verbatim from trial_runs\descriptor_utils\..\check_remove_non_common_ids.py
		#Note to self: Needed to make input and output file names absolute as CWD no longer = directory in which the following files are found
		file_one = r'%s\toy_fp_file_with_fp_for_all_mols.txt' % "\\".join(os.path.abspath(__file__).split('\\')[:-1])
		file_one_new_name = r'%s\toy_filtered_fp_file.txt' % "\\".join(os.path.abspath(__file__).split('\\')[:-1])
		file_one_delim = '\t'
		file_one_first_mol_line = 1
		
		file_two = r'%s\toy_NumericDescs_file_with_Descs_not_computed_for_all_mols.csv' % "\\".join(os.path.abspath(__file__).split('\\')[:-1])
		file_two_new_name = r'%s\toy_parsed_NumericDescs_file.csv' % "\\".join(os.path.abspath(__file__).split('\\')[:-1])
		file_two_delim=','
		file_two_first_mol_line = 2
		
		#from descriptor_utils import descriptorsFilesProcessor #Note to self: this line replaced with next!
		from ml_input_utils import descriptorsFilesProcessor
		
		my_descriptorsFilesProcessor = descriptorsFilesProcessor()
		
		my_descriptorsFilesProcessor.only_retain_entries_in_descriptors_files_corresponding_to_common_IDs(file_one,file_one_delim,file_one_first_mol_line,file_one_new_name,file_two,file_two_delim,file_two_first_mol_line,file_two_new_name)
		
		#Note to self: ABOVE copied verbatim from trial_runs\descriptor_utils\..\check_remove_non_common_ids.py
		
		del my_descriptorsFilesProcessor
		del descriptorsFilesProcessor
		
		all_input_files_required_for_unittesting = [file_one,file_two]
		
		all_orig_output_files_to_be_compared_as_required_for_unittesting = []
		for new_file in [file_one_new_name,file_two_new_name]:
			file_ext = new_file.split('.')[-1]
			orig_file = re.sub('(\.%s$)' % file_ext,' - Copy.%s' % file_ext,new_file)
			all_orig_output_files_to_be_compared_as_required_for_unittesting.append(orig_file)
			self.compareOriginalAndNewFiles(orig_file,new_file)
		
		files_not_to_delete = all_input_files_required_for_unittesting+all_orig_output_files_to_be_compared_as_required_for_unittesting
		self.clean_up_if_all_checks_passed(specific_files_not_to_delete=files_not_to_delete)
