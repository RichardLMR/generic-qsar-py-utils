#########################################################################################################
# ml_input_utils.py
# One of the Python modules written as part of the genericQSARpyUtils project (see below).
#
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
#<N.B.: All file name manipulation supposes this code is running under Windows!>

import re,os,itertools,sys,csv
from collections import defaultdict #Assumption: Python version >= 2.5
import functools
import pybel

class descriptorsFilesProcessor():
	def __init__(self):
		pass
	
	def match_ids_to_string_fp_features(self,string_fp_file,jCompoundMapperStringFeatures=False):
		id2string_fp_features = {} #N.B.: For now, we will only compute binary descriptors based upon feature occurence => only the set of unique features per compound is required!
		f_in = open(string_fp_file)
		try:
			lines = [LINE.replace('\n','') for LINE in f_in.readlines()]
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
		
			feat2IndexFileName = re.sub('(\.txt$)','_fpFeat2InitialIndex.csv',raw_fp_file)#17/03/13: actually, it is useful to write this to the same directory as the fingerprints file! => Hopefully any associated errors can be dealt with!#.split("\\")[-1]) #16/01/2013, 15:25: this line was put back in - since unittests failed when it was replaced with the following line!
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
		
		f_out = open(descriptors_file_name,'w')
		try:
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
			all_data_lines = [LINE.replace('\n','') for LINE in f_in.readlines()]
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
			input_lines = [LINE.replace('\n','') for LINE in f_in.readlines()]
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

