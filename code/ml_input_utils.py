#########################################################################################################
# ml_input_utils.py
# One of the Python modules written as part of the genericQSARpyUtils project (see below).

# ################################################
# #ml_input_utils.py: Key documentation :Contents#
# ################################################
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
#<N.B.: All file name manipulation supposes this code is running under Windows!>

import re,os,itertools,sys,csv
from collections import defaultdict #Assumption: Python version >= 2.5
import functools
####################################################################################
#####<KEY GLOBAL VARIABLES REQUIRED SPECIFICALLY FOR COMPUTING DESCRIPTORS (BELOW)>#
####################################################################################
OB_DIR = r'C:\Program Files (x86)\OpenBabel-2.3.0'
try:
	import pybel
except ImportError:
	try:
		os.environ['PATH'] += r';%s' % OB_DIR 
	except NameError:
		import os
		os.environ['PATH'] += r';%s' % OB_DIR 
	import pybel

dependencies_dir = r'%s\dependencies' % '\\'.join(os.path.abspath(__file__).split("\\")[:-2])
sys.path.append(dependencies_dir)

####################################################################################
#####<KEY GLOBAL VARIABLES REQUIRED SPECIFICALLY FOR COMPUTING DESCRIPTORS (ABOVE)>#
####################################################################################



class descriptorsGenerator():
	
	#General comment: all IDs assigned to molecular fingerprints/descriptor vectors will correspond to the molecule title.
	
	def __init__(self,molecules_input_file,output_dir=r''):
		self.molecules_input_file = molecules_input_file
		self.output_dir = output_dir
	def convert_input(self,required_format,out_name=''):
		'''
		descriptorsGenerator.convert_input(required_format)
		Some code for descriptors generation may require input files in a particular format - hence this function.
		The format of the input file is automatically detected from its extension - e.g. dataset.smi
		'''
		input_format = self.molecules_input_file.split(".")[-1]
		
		if '' == out_name:
			out_name = r'%s\%s' % (self.output_dir,re.sub('(\.%s$)' % input_format,'_pybel.%s' % required_format,self.molecules_input_file.split("\\")[-1])) #15/01/13: need to cope with the possibility that molecules_input_file is an asolute name!
		out = pybel.Outputfile(required_format,out_name,overwrite=True)
		try:
			for mol in pybel.readfile(input_format,self.molecules_input_file):
				out.write(mol)
		finally:
			out.close()
			del out
			try:
				del mol
			except NameError: #input file might mistakenly contain no molecules that were read!
				print '!'*50
				print 'No molecules read from input!'
				print '!'*50
				sys.exit(1)
		self.molecules_input_file = out_name
	
	def write_title_as_sdf_ID(self,ID_FIELD='descriptorsFileID',out_name=''):
		input_format = self.molecules_input_file.split(".")[-1]
		
		if '' == out_name:
			out_name =  r'%s\%s' % (self.output_dir,re.sub('(\.%s$)' % input_format,'_IDs.sdf',self.molecules_input_file.split("\\")[-1])) #15/01/13: need to cope with the possibility that molecules_input_file is an asolute name!
		out = pybel.Outputfile('sdf',out_name,overwrite=True)
		try:
			for mol in pybel.readfile(input_format,self.molecules_input_file):
				mol.data[ID_FIELD] = mol.title
				out.write(mol)
		finally:
			out.close()
			del out
			try:
				del mol
			except NameError: #input file might mistakenly contain no molecules that were read!
				print '!'*50
				print 'No molecules read from input!'
				print '!'*50
				sys.exit(1)
		self.molecules_input_file = out_name
	
	def write_jCompoundMapper_fp_file(self,atom_type='DAYLIGHT_INVARIANT_RING',fp_type='ECFP',distance_cutoff_or_search_depth=2,output_format='STRING_PATTERNS',out_name=''):
		#<By default, this function should calculate ECFP_4 fingerprints as close as possible - excluding hashing! - to those described by Rogers and Hahn 2010>
		#<N.B.: The arguments for this function and functionArg2jCompoundMapperCommandLineArg (below) are based upon TutorialEnvTox.pdf, downloaded from http://jcompoundmapper.sourceforge.net/ on the 03/09/12>
		#N.B.: Contrary to tutorial, numbers do not seem to work?
		#N.B.: My impression is that the :1 added to the end of each jCompoundMapper (with default arguments specified above) feature is superfluous - and that this must be removed to give a valid SMARTS!
		###############################################
		functionArg2jCompoundMapperCommandLineArg = defaultdict(dict)
		functionArg2jCompoundMapperCommandLineArg['atom_type']['DAYLIGHT_INVARIANT_RING'] = '-a DAYLIGHT_INVARIANT_RING' 
		functionArg2jCompoundMapperCommandLineArg['atom_type']['DAYLIGHT_INVARIANT'] = '-a DAYLIGHT_INVARIANT'
		functionArg2jCompoundMapperCommandLineArg['atom_type']['CUSTOM'] = '-a CUSTOM'
		functionArg2jCompoundMapperCommandLineArg['atom_type']['ELEMENT_SYMBOL'] = '-a ELEMENT_SYMBOL'
		functionArg2jCompoundMapperCommandLineArg['atom_type']['ELEMENT_NEIGHBOR_RING'] = '-a ELEMENT_NEIGHBOR_RING'
		functionArg2jCompoundMapperCommandLineArg['atom_type']['ELEMENT_NEIGHBOR'] = '-a ELEMENT_NEIGHBOR'
		functionArg2jCompoundMapperCommandLineArg['atom_type']['CDK_ATOM_TYPES'] = '-a CDK_ATOM_TYPES'
		
		functionArg2jCompoundMapperCommandLineArg['fp_type']['RAD3D'] = '-c RAD3D'
		functionArg2jCompoundMapperCommandLineArg['fp_type']['RAD2D'] = '-c RAD2D'
		functionArg2jCompoundMapperCommandLineArg['fp_type']['SHED'] = '-c SHED'
		functionArg2jCompoundMapperCommandLineArg['fp_type']['LSTAR'] = '-c LSTAR'
		functionArg2jCompoundMapperCommandLineArg['fp_type']['ECFP'] = '-c ECFP'
		functionArg2jCompoundMapperCommandLineArg['fp_type']['PHAP3POINT3D'] = '-c PHAP3POINT3D'
		functionArg2jCompoundMapperCommandLineArg['fp_type']['PHAP2POINT3D'] = '-c PHAP2POINT3D'
		functionArg2jCompoundMapperCommandLineArg['fp_type']['PHAP3POINT2D'] = '-c PHAP3POINT2D'
		functionArg2jCompoundMapperCommandLineArg['fp_type']['PHAP2POINT2D'] = '-c PHAP2POINT2D'
		functionArg2jCompoundMapperCommandLineArg['fp_type']['CATS3D'] = '-c CATS3D'
		functionArg2jCompoundMapperCommandLineArg['fp_type']['CATS2D'] = '-c CATS2D'
		functionArg2jCompoundMapperCommandLineArg['fp_type']['AT3D'] = '-c AT3D'
		functionArg2jCompoundMapperCommandLineArg['fp_type']['AP3D'] = '-c AP3D'
		functionArg2jCompoundMapperCommandLineArg['fp_type']['AT2D'] = '-c AT2D'
		functionArg2jCompoundMapperCommandLineArg['fp_type']['AP2D'] = '-c AP2D'
		functionArg2jCompoundMapperCommandLineArg['fp_type']['ASP'] = '-c ASP'
		functionArg2jCompoundMapperCommandLineArg['fp_type']['DFS'] = '-c DFS'
		
		functionArg2jCompoundMapperCommandLineArg['output_format']['WEKA_HASHED'] = '-ff WEKA_HASHED'
		functionArg2jCompoundMapperCommandLineArg['output_format']['STRING_PATTERNS'] = '-ff STRING_PATTERNS'
		functionArg2jCompoundMapperCommandLineArg['output_format']['FULL_CSV'] = '-ff FULL_CSV'
		functionArg2jCompoundMapperCommandLineArg['output_format']['LIBSVM_MATRIX'] = '-ff LIBSVM_MATRIX'
		functionArg2jCompoundMapperCommandLineArg['output_format']['LIBSVM_SPARSE'] = '-ff LIBSVM_SPARSE'
		################################################
		
		#####################
		autoAssignOutputFileExtension = {}
		autoAssignOutputFileExtension['WEKA_HASHED'] ='hashed.arff' 
		autoAssignOutputFileExtension['STRING_PATTERNS'] ='txt'
		autoAssignOutputFileExtension['LIBSVM_MATRIX'] ='libsvm.matrix.dat' 
		autoAssignOutputFileExtension['LIBSVM_SPARSE'] ='libsvm.sparse.dat' 
		#####################
		
		####
		input_format = self.molecules_input_file.split(".")[-1]
		if not 'sdf' == input_format:
			self.convert_input('sdf')
		self.write_title_as_sdf_ID(ID_FIELD='descriptorsFileID')
		####
		
		cli_options = ' '.join([functionArg2jCompoundMapperCommandLineArg['atom_type'][atom_type],functionArg2jCompoundMapperCommandLineArg['fp_type'][fp_type],'-d %d' % distance_cutoff_or_search_depth,functionArg2jCompoundMapperCommandLineArg['output_format'][output_format]])
		
		if '' == out_name:
			out_name = r'%s\%s' % (self.output_dir,re.sub('(\.sdf$)','_%s.%s' % (re.sub('(\s+)','',cli_options),autoAssignOutputFileExtension[output_format]),self.molecules_input_file.split("\\")[-1]))
		
		cmd = r'java -jar %s\jCMapperCLI.jar -f %s %s -o %s' % (dependencies_dir,self.molecules_input_file,cli_options,out_name)
		
		assert 0 == os.system(cmd), r" This cmd failed: %s" % cmd
		
		return out_name
	
	def write_scaffolds_fp_file(self,scaffolds_fp_file=''):
		'''
		descriptorsGenerator.write_scaffolds_fp_file()
		Parses the current molecules inpur file being parsed by an instance of the descriptorsGenerator class.
		Aim: Determine which scaffold(s) is (are) present in these molecules, and write out the corresponding smiles patterns 
		present in each molecule in the form of a string fingerprint - i.e. the output file would contain lines of the following format:
		<molecule ID>\t<scaffold smiles string>....
		'''
		
		######
		#04/09/12: Currently using code from murcko.py downloaded (today) from http://flo.nigsch.com/?p=29
		#####
		
		from murcko import GetFusedRingsMatrix,GetFusedRings,GetAtomsInRingSystems,GetCanonicalFragments
		
		input_file_format = self.molecules_input_file.split(".")[-1]
		
		if '' == scaffolds_fp_file:
			scaffolds_fp_file = r'%s\%s' % (self.output_dir,re.sub('(\.%s$)' % input_file_format,'_scaffoldsFP.txt',self.molecules_input_file.split("\\")[-1]))
		
		
		
		f_out = open(scaffolds_fp_file,'w')
		try:
			for mol in pybel.readfile(input_file_format,self.molecules_input_file):
				
				mol.OBMol.DeleteHydrogens()
				smiles = mol.write("smi").split("\t")[0]
				canmol = pybel.readstring("smi", smiles)
				FusedRingsMatrix = GetFusedRingsMatrix(canmol)
				FusedRings = GetFusedRings(FusedRingsMatrix, len(canmol.sssr))
				RingSystems = GetAtomsInRingSystems(canmol, FusedRings, inclexo=True)
				frags = GetCanonicalFragments(smiles, RingSystems)
				
				f_out.write('\t'.join([mol.title]+frags)+'\n')
		finally:
			f_out.close()
			del f_out
		
		return scaffolds_fp_file



class descriptorsFilesProcessor():
	def __init__(self):
		pass
	
	##################
	#04/10/12: cut these methods (BELOW) from descriptorsGenerator() class
	##################
	
	def match_ids_to_string_fp_features(self,string_fp_file,jCompoundMapperStringFeatures=False):
		id2string_fp_features = {} #N.B.: For now, we will only compute binary descriptors based upon feature occurence => only the set of unique features per compound is required!
		f_in = open(string_fp_file)
		try:
			lines = [re.sub(r'\r|\n','',LINE) for LINE in f_in.readlines()]
			assert not 0 == len(lines), " Fingerprints file is empty???"
			del LINE
		finally:
			f_in.close()
			del f_in
		
		for LINE in lines:
			if jCompoundMapperStringFeatures:
				ID = re.sub('(_INDEX=[0-9]+)','',LINE.split('\t')[0])
				features = list(set([re.sub('(\:1$)','',raw_feat) for raw_feat in LINE.split('\t')[1:]]))
			else:
				ID = LINE.split('\t')[0]
				features = list(set([raw_feat for raw_feat in LINE.split('\t')[1:]]))
			features.sort() #15/01/13:new line inserted
			id2string_fp_features[ID] = features
		del LINE
		
		#assert len(id2string_fp_features) == len(lines), " Duplicate IDs???" #-Better handled within script body - can call utility function to identify which IDs are duplicated!
		
		return id2string_fp_features
	
	def match_all_unique_features_to_indices(self,id2features,feat2IndexFileName='feat2Index.csv'):
		feat2Exists = defaultdict(bool) #is this a faster way to get all unique features than simply building up a list and then applying list(set(built_up_list))?
		
		for id in id2features:
			for FEATURE in id2features[id]:
				feat2Exists[FEATURE] = True
		del id
		del FEATURE
		
		feat2Index = defaultdict(int) #values should default to zero - a pre-requisite for this function and convert_id2features_to_svm_light_format_descriptors_file(...)!
		
		#for FEATURE in feat2Exists.keys(): ###15/01/13: commented out
		features = feat2Exists.keys() #15/01/13:new line inserted
		features.sort() #15/01/13:new line inserted
		feat_count = 0 #15/01/13:new line inserted
		for FEATURE in features: #15/01/13:new line inserted
			#feat2Index[FEATURE] += range(1,len(feat2Exists.keys())+1)[feat2Exists.keys().index(FEATURE)] ###15/01/13: commented out
			feat_count += 1 #15/01/13:new line inserted
			feat2Index[FEATURE] = feat_count #15/01/13:new line inserted
		del FEATURE
		del feat_count #15/01/13:new line inserted
		#############################################################################################
		#Record the correspondence between features and indices for subsequent model intepretation###
		#############################################################################################
		f_out = open(feat2IndexFileName,'w')
		try:
			f_out.write('Feature(Quoted),Index\n') #Quoting should make it possible to inspect this file in Excel...
			for FEATURE in feat2Index:
				f_out.write('"%s",%d\n' % (FEATURE,feat2Index[FEATURE]))
		finally:
			f_out.close()
			del f_out
		#############################################################################################
		
		
		return feat2Index
	
	def convert_id2features_to_svm_light_format_descriptors_file(self,descriptors_file_name,id2features,id2responseVariable=defaultdict(int),allUniqueFeaturesMatched2Indices=None):
		######
		#Desirable if then wish to load dataset for Machine Learning using scikit-learn: http://scikit-learn.org/stable/datasets/index.html#datasets-in-svmlight-libsvm-format (03/09/12)
		####
		
		descriptors_file_name_format = descriptors_file_name.split(".")[-1]
		
		if allUniqueFeaturesMatched2Indices is None:
			
			feat2IndexFileName= re.sub('(\.%s$)' % descriptors_file_name_format,'_feat2Index.csv',descriptors_file_name)
			
			
			allUniqueFeaturesMatched2Indices = self.match_all_unique_features_to_indices(id2features,feat2IndexFileName)
		
		################################################################################################
		#Sort IDs to ensure corresponding instances are written in a known order to the output file#####
		#Record this order - to allow, hopefully, for poorly predicted instances etc. to be identified!#
		################################################################################################
		all_ids = [ID for ID in id2features]
		del ID
		all_ids.sort()
		
		#DEBUG:
		#print all_ids #As expected for trial example!
		#sys.exit(1)
		#####
		
		ids_record = re.sub('(\.%s$)' % descriptors_file_name_format,'_recordOfIDsOrder.csv',descriptors_file_name)
		f_out = open(ids_record,'w')
		try:
			f_out.write('ID\n')
			for ID in all_ids:
				f_out.write(ID+'\n')
		finally:
			f_out.close()
			del f_out
		del ID
		#################################################################################################
		
		del descriptors_file_name_format
		
		f_out = open(descriptors_file_name,'w')
		try:
			
			for ID in all_ids:
				current_line_list = []
				current_line_list += ['%s' % id2responseVariable[ID]]
				indices_of_present_features = [allUniqueFeaturesMatched2Indices[feature] for feature in id2features[ID] if allUniqueFeaturesMatched2Indices[feature]>0]
				indices_of_present_features.sort()
				current_line_list += ['%d:1' % INDEX for INDEX in indices_of_present_features] ####Necessary assumption: zero-valued descriptors can be ignored! ###This assumption seems to be confirmed here: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/datasets/svmlight_format.py (accessed 04/09/12)
				del indices_of_present_features
				f_out.write(' '.join(current_line_list)+'\n')
			del ID
			del current_line_list
		finally:
			f_out.close()
			del f_out
	
	def generate_molId2DescId2DescValue_from_raw_fp_file(self,raw_fp_file,iSjCompoundMapperStringFeatures=False,unique_features_file=None):
		'''
		generate_molId2DescId2DescValue_from_raw_fp_file(raw_fp_file,iSjCompoundMapperStringFeatures=False,unique_features_file=None)
		
		(1) raw_fp_file :
		Must have the following structure to each line:
		molId\tFeatureB\tFeatureC\tFeatureA\tFeatureX....
		
		Must - for now! - have a .txt extension!
		
		(2) unique_features_file :
		Must have the same format as feat2IndexFileName (see contents of self.match_all_unique_features_to_indices(...).
		
		'''
		id2string_fp_features = self.match_ids_to_string_fp_features(raw_fp_file,iSjCompoundMapperStringFeatures)
		
		if unique_features_file is None:
		
			feat2IndexFileName = re.sub('(\.txt$)','_fpFeat2InitialIndex.csv',raw_fp_file.split("\\")[-1]) #16/01/2013, 15:25: this line was put back in - since unittests failed when it was replaced with the following line!
			#feat2IndexFileName = re.sub('(\.txt$)','_fpFeat2InitialIndex.csv',raw_fp_file) 
		
			feat2Index = self.match_all_unique_features_to_indices(id2string_fp_features,feat2IndexFileName)
		
		else:
			feat2IndexFileName = unique_features_file
			
			feat2Index = {}
			
			f_in = open(unique_features_file)
			
			try:
				data = csv.DictReader(f_in)
				for LINE in data:
					feat2Index[re.sub('("$|^")','',LINE['Feature(Quoted)'])] = int(LINE['Index'])
				del LINE
				del data
			finally:
				f_in.close()
				del f_in
		
		molId2DescId2DescValue = defaultdict(functools.partial(defaultdict,int))
		
		for molId in id2string_fp_features:
			# ########################
			# ########Initialise######
			# ########################
			# for feat in feat2Index:
				# molId2DescId2DescValue[molId][feat2Index[feat]] = 0
			# del feat
			# ########################
			
			
			for feat in id2string_fp_features[molId]:
				molId2DescId2DescValue[molId][feat2Index[feat]] = 1
		
		return molId2DescId2DescValue, feat2IndexFileName #5/01/13: I think the problem (TypeError) arose because this must have been updated to not just return molId2DescId2DescValue, but forgot to update generate_molId2DescId2DescValue_from_multiple_descriptors_files(...) - see below.
	
	def generate_molId2DescId2DescValue_from_CSV(self,raw_descriptors_csv):
		'''
		generate_molId2DescId2DescValue_from_CSV(raw_descriptors_csv)
		raw_descriptors_csv - must have the following structure:
		First line = Header => "molID,<Descriptor1:Name>,<Descriptor2:Name>,..."
		Subsequent lines:
		molId,<Descriptor1:Value>,<Descriptor2:Value>,....
		'''
		
		molId2DescId2DescValue = defaultdict(functools.partial(defaultdict,int))
		
		f_in = open(raw_descriptors_csv)
		try:
			data = [LINE for LINE in csv.DictReader(f_in)]
			del LINE
			descriptor_names = [KEY_NAME for KEY_NAME in data[0].keys() if not 'molID'==KEY_NAME]
			descName2descId = dict(zip(descriptor_names,range(1,len(descriptor_names)+1)))
			del descriptor_names
			############################################################
			#First record the (current) descriptor name: descId pairing#
			############################################################
			f_out = open(re.sub('(\.csv$)','_descName2InitialDescId.csv',raw_descriptors_csv.split("\\")[-1]),'w')
			try:
				f_out.write('DescriptorName,InitId\n')
				for descName in descName2descId:
					f_out.write('"%s",%s\n' % (descName,descName2descId[descName]))
				del descName
			finally:
				f_out.close()
				del f_out
			############################################################
			for mol_line in data:
				for descName in descName2descId:
					molId2DescId2DescValue[mol_line['molID']][descName2descId[descName]] = float(mol_line[descName])
				del descName
			del mol_line
		finally:
			f_in.close()
			del f_in
			del data
		
		return molId2DescId2DescValue
	
	def generate_molId2DescId2DescValue_from_multiple_descriptors_files(self,list_of_descriptors_files,corresponding_list_of_whether_descriptors_file_is_actually_a_raw_fp_file,corresponding_list_of_whether_descriptors_file_is_actually_a_jCompoundMapperStringFeatures_file):
		
		assert len(list_of_descriptors_files) == len(set(list_of_descriptors_files)), " %s ???" % list_of_descriptors_files
		assert len(list_of_descriptors_files) == len(corresponding_list_of_whether_descriptors_file_is_actually_a_raw_fp_file) , " %d vs. %d ???" % (len(list_of_descriptors_files),len(corresponding_list_of_whether_descriptors_file_is_actually_a_raw_fp_file))
		assert len(list_of_descriptors_files) == len(corresponding_list_of_whether_descriptors_file_is_actually_a_jCompoundMapperStringFeatures_file), " %d vs. %d ???" % (len(list_of_descriptors_files),len(corresponding_list_of_whether_descriptors_file_is_actually_a_jCompoundMapperStringFeatures_file))
		
		#Clearly, all descriptors/raw fp files parsed MUST correspond to the same molecule IDs!
		
		combined_molId2DescId2DescValue = defaultdict(functools.partial(defaultdict,int))
		
		current_initial_descriptor_id = 1
		
		
		for raw_descriptors_file in list_of_descriptors_files:
			if corresponding_list_of_whether_descriptors_file_is_actually_a_raw_fp_file[list_of_descriptors_files.index(raw_descriptors_file)]:
				iSjCompoundMapperStringFeatures = corresponding_list_of_whether_descriptors_file_is_actually_a_jCompoundMapperStringFeatures_file[list_of_descriptors_files.index(raw_descriptors_file)]
				current_molId2DescId2DescValue = self.generate_molId2DescId2DescValue_from_raw_fp_file(raw_descriptors_file,iSjCompoundMapperStringFeatures)[0] #15/01/2013: Now this line updated as required (see above).
			else:
				current_molId2DescId2DescValue = self.generate_molId2DescId2DescValue_from_CSV(raw_descriptors_file)
			
			all_current_original_desc_ids = []
			
			for molId in current_molId2DescId2DescValue:
				try: #15/01/2013: TypeError occured in next line when writing test_1.py based on earlier trial run of code!
					for descId in current_molId2DescId2DescValue[molId]:
					
						all_current_original_desc_ids.append(descId)
					
						combined_molId2DescId2DescValue[molId][(current_initial_descriptor_id-1)+descId] = float(current_molId2DescId2DescValue[molId][descId])
				except TypeError,err_msg:
					print '='*50
					print 'Following TypeError exception caught:\n ', err_msg
					print 'Problem occured when parsing:\n ', raw_descriptors_file
					print 'Corresponding current_molId2DescId2DescValue = \n' , current_molId2DescId2DescValue
					print '='*50
					sys.exit(1)
				
				del descId
			del molId
			
			all_current_original_desc_ids = list(set(all_current_original_desc_ids))
			
			current_initial_descriptor_id += len(all_current_original_desc_ids)
		del all_current_original_desc_ids
		del current_initial_descriptor_id
		
		#########################
		all_desc_ids = list(itertools.chain(*[combined_molId2DescId2DescValue[mol_ID].keys() for mol_ID in combined_molId2DescId2DescValue.keys()])) ####No keys assigned for zero valued FP descriptors!
		del mol_ID
		
		for molId in combined_molId2DescId2DescValue:
			for descId in all_desc_ids:
				combined_molId2DescId2DescValue[molId][descId] += 0.0
			del descId
		del molId
		#######################
		
		return combined_molId2DescId2DescValue
	
	def write_svmlight_format_modellingFile_for_generic_descriptors(self,molId2DescId2DescValue,descriptors_file_name,id2responseVariable=defaultdict(int)): 
		
		
		################################################################################################
		descriptors_file_name_format = descriptors_file_name.split(".")[-1]
		################################################################################################
		#Sort IDs to ensure corresponding instances are written in a known order to the output file#####
		#Record this order - to allow, hopefully, for poorly predicted instances etc. to be identified!#
		################################################################################################
		all_ids = [ID for ID in molId2DescId2DescValue]
		del ID
		all_ids.sort()
		
		#DEBUG:
		#print all_ids #As expected for trial example!
		#sys.exit(1)
		#####
		
		ids_record = re.sub('(\.%s$)' % descriptors_file_name_format,'_recordOfIDsOrder.txt',descriptors_file_name)
		f_out = open(ids_record,'w')
		try:
			f_out.write('ID\n')
			for ID in all_ids:
				f_out.write(ID+'\n')
		finally:
			f_out.close()
			del f_out
		del ID
		#################################################################################################
		del descriptors_file_name_format
		#################################################################################################
		
		all_desc_ids = molId2DescId2DescValue[all_ids[0]].keys()
		all_desc_ids.sort()
		
		f_out = open(descriptors_file_name,'w')
		try:
			for ID in all_ids:
				current_line_list = []
				current_line_list += ['%s' % id2responseVariable[ID]]
				
				current_line_list += ['%d:%f' %(descId,molId2DescId2DescValue[ID][descId]) for descId in all_desc_ids if not 0.0==molId2DescId2DescValue[ID][descId]] ####Necessary assumption: zero-valued descriptors can be ignored! ###This assumption seems to be confirmed here: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/datasets/svmlight_format.py (accessed 04/09/12)
				f_out.write(' '.join(current_line_list)+'\n')
			del ID
			del current_line_list
		finally:
			f_out.close()
			del f_out
		
		
	def write_csv_format_modellingFile_for_generic_descriptors(self,molId2DescId2DescValue,descriptors_file_name,id2responseVariable=defaultdict(int)):
		
		all_mol_ids = molId2DescId2DescValue.keys()
		all_mol_ids.sort()
		
		all_desc_ids = molId2DescId2DescValue[all_mol_ids[0]].keys()
		all_desc_ids.sort()
		
		f_out = open(descriptors_file_name,'w')
		try:
			header = ','.join(['molID','yValue']+['d%d' % descID for descID in all_desc_ids])
			del descID
			f_out.write(header+'\n')
			del header
			
			for molID in all_mol_ids:
				current_line_list = ['%s' % molID]
				current_line_list += ['%s' % id2responseVariable[molID]]
				current_line_list += ['%f' % molId2DescId2DescValue[molID][descId] for descId in all_desc_ids]
				del descId
				f_out.write(','.join(current_line_list)+'\n')
			del molID
			del current_line_list
		finally:
			f_out.close()
			del f_out
	
	##################
	#04/10/12: cut these methods (ABOVE) from descriptorsGenerator() class
	##################
	
	def only_retain_entries_in_descriptors_files_corresponding_to_common_IDs(self,file_one,file_one_delim,file_one_first_mol_line,file_one_new_name,file_two,file_two_delim,file_two_first_mol_line,file_two_new_name,file_one_IDs_position=0,file_two_IDs_position=0):
		################
		#It is possible that it may not have been possible to compute one set of descriptors for all molecules in a given dataset. If combining / comparing with different descriptor sets,those files based upon other sets of descriptors, computed for the same molecules, for which no such problems were encountered, need to be filtered accordingly.
		#<WARNING: This function would not work if any molIDs contained the specified delimiter!>
		#11/10/12:These new arguments were added: file_one_IDs_position,file_two_IDs_position
		################
		
		#---------------------
		f_in = open(file_one)
		try:
			file_one_all_lines = [re.sub(r'\r|\n','',LINE) for LINE in f_in.readlines()]
			del LINE
		finally:
			f_in.close()
			del f_in
		
		file_one_header_lines = file_one_all_lines[:(file_one_first_mol_line-1)]
		file_one_molecule_lines = file_one_all_lines[(file_one_first_mol_line-1):]
		del file_one_all_lines
		
		file_one_molID2MoleculeLine = {}
		for file_one_MOLECULE_LINE in file_one_molecule_lines:
			file_one_molID2MoleculeLine[re.sub('(_INDEX=[0-9]+)','',file_one_MOLECULE_LINE.split(file_one_delim)[file_one_IDs_position])] = file_one_MOLECULE_LINE #re.sub is just in case the file was generated using jCompoundMapper - which adds _INDEX=<NUMBER> onto the end of molecule IDs!
		
		del file_one_molecule_lines
		del file_one_MOLECULE_LINE
		#---------------------
		
		#---------------------
		f_in = open(file_two)
		try:
			file_two_all_lines = [re.sub(r'\r|\n','',LINE) for LINE in f_in.readlines()]
			del LINE
		finally:
			f_in.close()
			del f_in
		
		file_two_header_lines = file_two_all_lines[:(file_two_first_mol_line-1)]
		file_two_molecule_lines = file_two_all_lines[(file_two_first_mol_line-1):]
		del file_two_all_lines
		
		file_two_molID2MoleculeLine = {}
		for file_two_MOLECULE_LINE in file_two_molecule_lines:
			file_two_molID2MoleculeLine[re.sub('(_INDEX=[0-9]+)','',file_two_MOLECULE_LINE.split(file_two_delim)[file_two_IDs_position])] = file_two_MOLECULE_LINE #re.sub is just in case the file was generated using jCompoundMapper - which adds _INDEX=<NUMBER> onto the end of molecule IDs!
		
		del file_two_molecule_lines
		del file_two_MOLECULE_LINE
		#---------------------
		
		molIDsInOneNotTwo = list(set(file_one_molID2MoleculeLine.keys()).difference(set(file_two_molID2MoleculeLine.keys())))
		
		molIDsInTwoNotOne = list(set(file_two_molID2MoleculeLine.keys()).difference(set(file_one_molID2MoleculeLine.keys())))
		
		common_molIDs = list(set(file_two_molID2MoleculeLine.keys()).intersection(set(file_one_molID2MoleculeLine.keys())))
		
		common_molIDs.sort() ####12/10/12: Update: Otherwise, even if an input file's only IDs corresponded exactly to the common IDs, the file contents could still be changed by virtue of re-ordering the IDs...; this sorting would fix that problem *if* the input file's IDs were sorted in this fashion as well!!!
		
		print '-'*50
		print 'Molecule IDs in ', file_one, ' not in ', file_two, ' :'
		for molID in molIDsInOneNotTwo:
			print molID
		if not 0 == len(molIDsInOneNotTwo):
			del molID
		print 'Total: %d.' % len(molIDsInOneNotTwo)
		print '-'*50
		
		print '-'*50
		print 'Molecule IDs in ', file_two, ' not in ', file_one, ' :'
		for molID in molIDsInTwoNotOne:
			print molID
		if not 0 == len(molIDsInTwoNotOne):
			del molID
		print 'Total: %d.' % len(molIDsInTwoNotOne)
		print '-'*50
		
		print '-'*50
		print '%d molecule IDs in common.' % len(common_molIDs)
		print '-'*50
		
		#-------------------------------
		print 'Writing out lines from ', file_one, ' corresponding to common molecule IDs to: ', file_one_new_name
		f_out = open(file_one_new_name,'w')
		try:
			for file_one_HEADER_LINE in file_one_header_lines:
				f_out.write(file_one_HEADER_LINE+'\n')
			for COMMON_ID in common_molIDs:
				f_out.write(file_one_molID2MoleculeLine[COMMON_ID]+'\n')
			del COMMON_ID
		finally:
			f_out.close()
			del f_out
		#--------------------------------
		
		#-------------------------------
		print 'Writing out lines from ', file_two, ' corresponding to common molecule IDs to: ', file_two_new_name
		f_out = open(file_two_new_name,'w')
		try:
			for file_two_HEADER_LINE in file_two_header_lines:
				f_out.write(file_two_HEADER_LINE+'\n')
			for COMMON_ID in common_molIDs:
				f_out.write(file_two_molID2MoleculeLine[COMMON_ID]+'\n')
		finally:
			f_out.close()
			del f_out
		#--------------------------------
	
	def write_csv_format_modellingFile_from_multiple_descriptors_files(self,list_of_descriptors_files,corresponding_list_of_whether_descriptors_file_is_actually_a_raw_fp_file,corresponding_list_of_whether_descriptors_file_is_actually_a_jCompoundMapperStringFeatures_file,descriptors_file_name,id2responseVariable=defaultdict(int),corresponding_list_of_unique_features_files=[None]):
		#####
		#<10/10/12: TO DO: CHECK FUNCTION WORKS GIVEN NEW UPDATES (CORRESPONDING TO NEW ARGUMENT: corresponding_list_of_unique_features_files) ALLOWING FOR FINGERPRINT FEATURES DEFINING DESCRIPTORS TO BE PRE-SPECIFIED!>
		#####
		##########################################################################################
		#<N.B.: When trying to use generate_molId2DescId2DescValue_from_multiple_descriptors_files(...) followed by write_csv_format_modellingFile_for_generic_descriptors(...) for a dataset of > 5,000 molecules with a high dimensional, sparse descriptor set, a MemoryError exception was thrown. This function aims to dynamically write descriptor vectors to file at the point at which generate_molId2DescId2DescValue_from_multiple_descriptors_files(...) threw this exception.>
		###########################################################################################
		
		
		
		assert len(list_of_descriptors_files) == len(set(list_of_descriptors_files)), " %s ???" % list_of_descriptors_files
		assert len(list_of_descriptors_files) == len(corresponding_list_of_whether_descriptors_file_is_actually_a_raw_fp_file) , " %d vs. %d ???" % (len(list_of_descriptors_files),len(corresponding_list_of_whether_descriptors_file_is_actually_a_raw_fp_file))
		assert len(list_of_descriptors_files) == len(corresponding_list_of_whether_descriptors_file_is_actually_a_jCompoundMapperStringFeatures_file), " %d vs. %d ???" % (len(list_of_descriptors_files),len(corresponding_list_of_whether_descriptors_file_is_actually_a_jCompoundMapperStringFeatures_file))
		
		if [None] == corresponding_list_of_unique_features_files:
			corresponding_list_of_unique_features_files = corresponding_list_of_unique_features_files*len(list_of_descriptors_files)
			record_of_all_feat2IndexFiles = []
		else:
			record_of_all_feat2IndexFiles = [None]*len(list_of_descriptors_files)
		
		#Clearly, all descriptors/raw fp files parsed MUST correspond to the same molecule IDs!
		
		combined_molId2DescId2DescValue = defaultdict(functools.partial(defaultdict,int))
		
		current_initial_descriptor_id = 1
		
		
		for raw_descriptors_file in list_of_descriptors_files:
			if corresponding_list_of_whether_descriptors_file_is_actually_a_raw_fp_file[list_of_descriptors_files.index(raw_descriptors_file)]:
				
				iSjCompoundMapperStringFeatures = corresponding_list_of_whether_descriptors_file_is_actually_a_jCompoundMapperStringFeatures_file[list_of_descriptors_files.index(raw_descriptors_file)]
				
				unique_features_file = corresponding_list_of_unique_features_files[list_of_descriptors_files.index(raw_descriptors_file)]
				
				current_molId2DescId2DescValue, feat2IndexFile = self.generate_molId2DescId2DescValue_from_raw_fp_file(raw_descriptors_file,iSjCompoundMapperStringFeatures,unique_features_file)
				
				if unique_features_file is None:
					record_of_all_feat2IndexFiles.append(feat2IndexFile)
				else:
					assert feat2IndexFile == unique_features_file
				
			else:
				current_molId2DescId2DescValue = self.generate_molId2DescId2DescValue_from_CSV(raw_descriptors_file)
			
			all_current_original_desc_ids = []
			
			for molId in current_molId2DescId2DescValue:
				for descId in current_molId2DescId2DescValue[molId]:
					
					all_current_original_desc_ids.append(descId)
					
					combined_molId2DescId2DescValue[molId][(current_initial_descriptor_id-1)+descId] = float(current_molId2DescId2DescValue[molId][descId])
				
				del descId
			del molId
			
			all_current_original_desc_ids = list(set(all_current_original_desc_ids))
			
			current_initial_descriptor_id += len(all_current_original_desc_ids)
		del all_current_original_desc_ids
		del current_initial_descriptor_id
		
		#########################
		all_desc_ids = list(set(list(itertools.chain(*[combined_molId2DescId2DescValue[mol_ID].keys() for mol_ID in combined_molId2DescId2DescValue.keys()])))) ####No keys assigned for zero valued FP descriptors!
		del mol_ID
		all_desc_ids.sort()
		
		# for molId in combined_molId2DescId2DescValue:
			# for descId in all_desc_ids:
				# combined_molId2DescId2DescValue[molId][descId] += 0.0 #<N.B.>: This was where a MemoryError exception was thrown before
			# del descId
		# del molId
		
		
		f_out = open(descriptors_file_name,'w')
		try:
			header = ','.join(['molID','yValue']+['d%d' % descID for descID in all_desc_ids])
			del descID
			f_out.write(header+'\n')
			del header
			
			all_mol_ids = combined_molId2DescId2DescValue.keys()
			
			#####################################################################################################
			#N.B.: Should ensure (i.e. to make sure selection of the same rows, e.g. for a train/test partition or when doing bootstrapping) that  substances are written to the model input file in the same order irrespective of the descriptors set used for modelling!
			#This will be taken care of by sorting the IDs prior to writing the corresponding entries to the output file.
			#####################################################################################################
			
			all_mol_ids.sort()
			
			for molID in all_mol_ids:
				current_line_list = ['%s' % molID]
				current_line_list += ['%s' % id2responseVariable[molID]]
				#current_line_list += ['%f' % molId2DescId2DescValue[molID][descId] for descId in all_desc_ids]
				
				################################################################
				#Hopefully this will avoid a MemoryError exception!#############
				################################################################
				current_DescId2DescValue = combined_molId2DescId2DescValue[molID]
				
				del combined_molId2DescId2DescValue[molID]
				
				for descId in all_desc_ids:
					current_DescId2DescValue[descId] += 0.0
				del descId
				
				current_line_list += ['%f' % current_DescId2DescValue[descId] for descId in all_desc_ids]
				del descId
				del current_DescId2DescValue
				
				#################################################################
				
				f_out.write(','.join(current_line_list)+'\n')
			del molID
			del current_line_list
		finally:
			f_out.close()
			del f_out
		
		#######################
		
		return record_of_all_feat2IndexFiles
	
	
	def write_svmlight_format_modellingFile_from_multiple_descriptors_files(self,list_of_descriptors_files,corresponding_list_of_whether_descriptors_file_is_actually_a_raw_fp_file,corresponding_list_of_whether_descriptors_file_is_actually_a_jCompoundMapperStringFeatures_file,descriptors_file_name,id2responseVariable=defaultdict(int),corresponding_list_of_unique_features_files=[None]):
		#p.t.r.d.i.:DONE
		#####################################################################################################
		#<N.B.: 09/10/12: Adapted from write_csv_format_modellingFile_from_multiple_descriptors_files(...).>
		#<10/10/12: But, unlike the OLD version of write_csv_format_modellingFile_from_multiple_descriptors_files(...), the possibility of defining the descriptors for fingerprint features files ('fp files') based upon an externally specified set of unique features has been introduced via the new argument: corresponding_list_of_unique_features_files!>
		#####################################################################################################
		
		
		
		assert len(list_of_descriptors_files) == len(set(list_of_descriptors_files)), " %s ???" % list_of_descriptors_files
		assert len(list_of_descriptors_files) == len(corresponding_list_of_whether_descriptors_file_is_actually_a_raw_fp_file) , " %d vs. %d ???" % (len(list_of_descriptors_files),len(corresponding_list_of_whether_descriptors_file_is_actually_a_raw_fp_file))
		assert len(list_of_descriptors_files) == len(corresponding_list_of_whether_descriptors_file_is_actually_a_jCompoundMapperStringFeatures_file), " %d vs. %d ???" % (len(list_of_descriptors_files),len(corresponding_list_of_whether_descriptors_file_is_actually_a_jCompoundMapperStringFeatures_file))
		
		if [None] == corresponding_list_of_unique_features_files:
			corresponding_list_of_unique_features_files = corresponding_list_of_unique_features_files*len(list_of_descriptors_files)
			record_of_all_feat2IndexFiles = []
		else:
			record_of_all_feat2IndexFiles = [None]*len(list_of_descriptors_files)
		
		#Clearly, all descriptors/raw fp files parsed MUST correspond to the same molecule IDs!
		
		combined_molId2DescId2DescValue = defaultdict(functools.partial(defaultdict,int))
		
		current_initial_descriptor_id = 1
		
		
		
		for raw_descriptors_file in list_of_descriptors_files:
			if corresponding_list_of_whether_descriptors_file_is_actually_a_raw_fp_file[list_of_descriptors_files.index(raw_descriptors_file)]:
				
				iSjCompoundMapperStringFeatures = corresponding_list_of_whether_descriptors_file_is_actually_a_jCompoundMapperStringFeatures_file[list_of_descriptors_files.index(raw_descriptors_file)]
				
				unique_features_file = corresponding_list_of_unique_features_files[list_of_descriptors_files.index(raw_descriptors_file)]
				
				current_molId2DescId2DescValue, feat2IndexFile = self.generate_molId2DescId2DescValue_from_raw_fp_file(raw_descriptors_file,iSjCompoundMapperStringFeatures,unique_features_file)
				
				if unique_features_file is None:
					record_of_all_feat2IndexFiles.append(feat2IndexFile)
				else:
					assert feat2IndexFile == unique_features_file
			else:
				current_molId2DescId2DescValue = self.generate_molId2DescId2DescValue_from_CSV(raw_descriptors_file)
			
			all_current_original_desc_ids = []
			
			for molId in current_molId2DescId2DescValue:
				for descId in current_molId2DescId2DescValue[molId]:
					
					all_current_original_desc_ids.append(descId)
					
					combined_molId2DescId2DescValue[molId][(current_initial_descriptor_id-1)+descId] = float(current_molId2DescId2DescValue[molId][descId])
				
				del descId
			del molId
			
			all_current_original_desc_ids = list(set(all_current_original_desc_ids))
			
			current_initial_descriptor_id += len(all_current_original_desc_ids)
		del all_current_original_desc_ids
		del current_initial_descriptor_id
		
		#########################
		all_desc_ids = list(set(list(itertools.chain(*[combined_molId2DescId2DescValue[mol_ID].keys() for mol_ID in combined_molId2DescId2DescValue.keys()])))) ####No keys assigned for zero valued FP descriptors!
		del mol_ID
		all_desc_ids.sort()
		
		# for molId in combined_molId2DescId2DescValue:
			# for descId in all_desc_ids:
				# combined_molId2DescId2DescValue[molId][descId] += 0.0 #<N.B.>: This was where a MemoryError exception was thrown before
			# del descId
		# del molId
		
		
		f_out = open(descriptors_file_name,'w')
		try:
			#header = ','.join(['molID','yValue']+['d%d' % descID for descID in all_desc_ids])
			#del descID
			#f_out.write(header+'\n')
			#del header
			
			all_mol_ids = combined_molId2DescId2DescValue.keys()
			
			#####################################################################################################
			#N.B.: Should ensure (i.e. to make sure selection of the same rows, e.g. for a train/test partition or when doing bootstrapping) that  substances are written to the model input file in the same order irrespective of the descriptors set used for modelling!
			#This will be taken care of by sorting the IDs prior to writing the corresponding entries to the output file.
			#####################################################################################################
			
			all_mol_ids.sort()
			
			for molID in all_mol_ids:
				current_line_list = ['%s' % id2responseVariable[molID]]
				
				
				################################################################
				#Hopefully this will avoid a MemoryError exception!#############
				################################################################
				current_DescId2DescValue = combined_molId2DescId2DescValue[molID]
				
				del combined_molId2DescId2DescValue[molID]
				
				# for descId in all_desc_ids:
					# current_DescId2DescValue[descId] += 0.0
				# del descId
				
				current_line_list += ['%d:%f' % (descId,current_DescId2DescValue[descId]) for descId in all_desc_ids if not 0.0 == current_DescId2DescValue[descId]]
				del descId
				del current_DescId2DescValue
				
				#################################################################
				
				f_out.write(' '.join(current_line_list)+'#%s' % molID+'\n') #svmlight format: anything following # should not be read into memory by correct parsers of this format!
			del molID
			del current_line_list
		finally:
			f_out.close()
			del f_out
		
		#######################
		
		return record_of_all_feat2IndexFiles
	
	def convert_svmlight_to_csv(self,svmlight_file,csv_file=r''):
		#d.i.p.t.r.:<DONE>
		molID2descID2Value = defaultdict(functools.partial(defaultdict,int))
		molID2responseValue = {}
		
		f_in = open(svmlight_file)
		try:
			all_data_lines = [re.sub(r'\r|\n','',LINE) for LINE in f_in.readlines()]
			del LINE
		finally:
			f_in.close()
			del f_in
		
		for LINE in all_data_lines:
			
			response_value_THEN_feature_ID_Value_Pairs, molID = LINE.split('#')
			
			response_value  = float(response_value_THEN_feature_ID_Value_Pairs.split()[0])
			
			molID2responseValue[molID] = response_value
			
			del response_value
			
			for feature_ID_Value_PAIR in response_value_THEN_feature_ID_Value_Pairs.split()[1:]:
				molID2descID2Value[molID][int(feature_ID_Value_PAIR.split(':')[0])] = float(feature_ID_Value_PAIR.split(':')[1])
			del response_value_THEN_feature_ID_Value_Pairs
			#del feature_ID_Value_PAIR ##Would fail if the current line corresponded to a molecule with no non-zero valued descriptors!
			del molID
		
		del LINE
		
		all_desc_ids = list(set(list(itertools.chain(*[molID2descID2Value[molID].keys() for molID in molID2descID2Value]))))
		all_desc_ids.sort()
		del molID
		
		if '' == csv_file:
			csv_file = re.sub('(\.%s$)' % svmlight_file.split('.')[-1] , '.csv',svmlight_file)
		
		f_out = open(csv_file,'w')
		try:
			#Copied (below) from above:
			header = ','.join(['molID','yValue']+['d%d' % descID for descID in all_desc_ids])
			del descID
			f_out.write(header+'\n')
			del header
			
			###########################
			
			all_mol_ids = molID2responseValue.keys() ####<***N.B.: If we select molecule IDs from molID2descID2Value.keys(), we would miss molecules with no non-zero valued descriptors!***><***TO DO: Fix this *possible* problem when generating initial svmlight/csv model input files in the methods of the current class presented above!****> 
			
			#Copied (below) from above:
			
			#####################################################################################################
			#N.B.: Should ensure (i.e. to make sure selection of the same rows, e.g. for a train/test partition or when doing bootstrapping) that  substances are written to the model input file in the same order irrespective of the descriptors set used for modelling!
			#This will be taken care of by sorting the IDs prior to writing the corresponding entries to the output file.
			#####################################################################################################
			
			all_mol_ids.sort()
			
			############################
			
			for molID in all_mol_ids:
				
				current_descID2Value = molID2descID2Value[molID]
				
				del molID2descID2Value[molID]
				
				for descID in all_desc_ids:
					
					current_descID2Value[descID] += 0.0
				
				del descID
				
				f_out.write(','.join([str(molID),str(molID2responseValue[molID])]+['%f' % current_descID2Value[descID] for descID in all_desc_ids])+'\n')
				
				del current_descID2Value
				
		finally:
			f_out.close()
			del f_out
		
		return csv_file
	
	def remove_response_values_column(self,ID_responseValue_descriptors_File,ID_descriptors_File='',responseValueColumnPosition=1,columnDelimiter=','):
		#d.i.p.t.r.:<DONE>
		f_in = open(ID_responseValue_descriptors_File)
		try:
			input_lines = [re.sub(r'\r|\n','',LINE) for LINE in f_in.readlines()]
			del LINE
		finally:
			f_in.close()
			del f_in
		
		###
		if '' == ID_descriptors_File:
			ID_descriptors_File = re.sub('(\.%s$)' % ID_responseValue_descriptors_File.split('.')[-1], '_noY.%s' % ID_responseValue_descriptors_File.split('.')[-1],ID_responseValue_descriptors_File)
		###
		
		f_out = open(ID_descriptors_File,'w')
		try:
			for LINE in input_lines:
				NEW_LINE = columnDelimiter.join([LINE.split(columnDelimiter)[col_pos] for col_pos in range(0,len(LINE.split(columnDelimiter))) if not col_pos ==  responseValueColumnPosition])
				f_out.write(NEW_LINE+'\n')
		finally:
			f_out.close()
			del f_out
		
		return ID_descriptors_File

