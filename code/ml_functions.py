#########################################################################################################
# ml_functions.py
# One of the Python modules written as part of the genericQSARpyUtils project (see below).

# ##############################################
# #ml_functions.py: Key documentation :Contents#
# ##############################################
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
import numpy as np
import re,os
from collections import defaultdict
import sklearn

def chi2(X,Y):
	return sklearn.feature_selection.chi2(X,Y)

def renameFilteredFeaturesFile(original_file,name_of_feature_selection_method,number_of_features_retained):
	file_format = original_file.split('.')[-1]
	new_file_name = re.sub('(\.%s$)' % file_format,'_fs_%s_top_%d.%s' % (name_of_feature_selection_method,number_of_features_retained,file_format),original_file)
	return new_file_name

def report_top_K_features(X_train_data,Y_train_data,univariate_scoring_function=chi2,number_of_features_to_retain=200):
	from sklearn.feature_selection import SelectKBest
	featureSelector = SelectKBest(score_func=univariate_scoring_function,k=number_of_features_to_retain)
	
	
	print '-'*50
	print 'Determining the top %d features, based on the training set information, according to this univariate scoring function: %s.' % (number_of_features_to_retain,univariate_scoring_function.__name__)
	featureSelector.fit(X_train_data,Y_train_data)
	print '-'*50
	
	return [1+zero_based_index for zero_based_index in list(featureSelector.get_support(indices=True))]


def filter_svmlight_format_line(LINE_WITHOUT_LINE_ENDING,indices_of_features_to_retain):
	class_then_descriptors_info, ID = LINE_WITHOUT_LINE_ENDING.split('#') 
	
	class_label = class_then_descriptors_info.split()[0]
	indexValuePairs = class_then_descriptors_info.split()[1:]
	
	del class_then_descriptors_info
	
	indexValuePairsToRetain = [indexValuePAIR for indexValuePAIR in indexValuePairs if int(indexValuePAIR.split(':')[0]) in indices_of_features_to_retain]
	
	new_LINE = '%s ' % class_label+' '.join(indexValuePairsToRetain)+'#%s' % ID
	
	return new_LINE

def remove_extra_features(orig_svmlight_format_file,new_svmlight_format_file,indices_of_features_to_retain):
	
	f_in = open(orig_svmlight_format_file)
	try:
		orig_data = [re.sub(r'\r|\n','',LINE) for LINE in f_in.readlines()]
		del LINE
	finally:
		f_in.close()
		del f_in
		
	
	f_out = open(new_svmlight_format_file,'w')
	try:
		for ORIG_LINE in orig_data:
			f_out.write(filter_svmlight_format_line(ORIG_LINE,indices_of_features_to_retain)+'\n')
	finally:
		f_out.close()
		del f_out

def filter_features_for_svmlight_format_files(svmlight_format_train_file,svmlight_format_test_file=None,univariate_scoring_function=chi2,number_of_features_to_retain=200):
	###################################################
	#d.i.p.t.r.(including called functions): <DONE>####
	###################################################
	'''
	This function can write out svmlight format versions of the training set *and*, if a corresponding test set file is specified, the test set, based only on the features selected using information from the training set.
	<*N.B.*: Only certain unvariate_scoring_function values will give sensible results for classification/regression datasets!
	-e.g. chi2 is appropriate for classification datasets.>
	'''
	from sklearn.datasets import load_svmlight_file
	
	
	
	###################
	####Read in files##
	###################
	
	print '-'*50
	print 'Reading this training set: ', svmlight_format_train_file
	X_train_data, Y_train_data = load_svmlight_file(svmlight_format_train_file)
	print '-'*50
	
	##################
	
	indices_of_top_K_features = report_top_K_features(X_train_data,Y_train_data,univariate_scoring_function,number_of_features_to_retain)
	
	print '='*50
	print 'Selected the features (i.e. descriptors) corresponding to these indices: '
	print indices_of_top_K_features ###<N.B.: It appears that these indices do not directly correspond to the indices of the features in the training and test set! - i.e. for the test example considered feature index 0 was selected, even though indices of features in the files parsed by this function started from 1!>
	print '='*50
	
	#######################
	#Writing output files##
	#######################
	
	output_files = {} #N.B.: Added after original test of this function [but no other changes made].
	
	for TRAIN_OR_TEST_LABEL in ['Train','Test']:
		if 'Train' == TRAIN_OR_TEST_LABEL:
			orig_svmlight_format_file = svmlight_format_train_file
		else:
			assert 'Test' == TRAIN_OR_TEST_LABEL
			orig_svmlight_format_file = svmlight_format_test_file
		
		if orig_svmlight_format_file is None:
			assert 'Test' == TRAIN_OR_TEST_LABEL , " orig_svmlight_format_file (for the training set) is deemed to be None???"
			continue
		
		new_svmlight_format_file = renameFilteredFeaturesFile(original_file=orig_svmlight_format_file,name_of_feature_selection_method=univariate_scoring_function.__name__,number_of_features_retained=number_of_features_to_retain)
		
		output_files[TRAIN_OR_TEST_LABEL] = new_svmlight_format_file
		
		remove_extra_features(orig_svmlight_format_file,new_svmlight_format_file,indices_of_top_K_features)
	
	#######################
	
	return output_files, indices_of_top_K_features #It may be interesting to analyse those features which were consistently selected across multiple train:test partitions in future work!


def mccv_svmlight_file(svmlight_file,output_dir=os.getcwd(),perc_test=0.2,repetitions=1,stratified=True,rng_seed=0):
	
	#d.i.p.t.r.(including called functions if applicable):<DONE>
	print '='*50
	print 'Carrying out Monte-Carlo cross-validation on %s - and writing all resultant train:test pairs to new files.' % svmlight_file
	print 'Fraction of data selected for testing: ', perc_test
	print 'Number of repetitions: ', repetitions
	print 'Stratified: %s.' % stratified
	
	
	
	f_in = open(svmlight_file)
	try:
		all_data_lines = [re.sub(r'\r|\n','',LINE) for LINE in f_in.readlines()]
		del LINE
	finally:
		f_in.close()
		del f_in
	
	if stratified:
		from sklearn.cross_validation import StratifiedShuffleSplit
		response_variables = np.array([float(LINE.split()[0]) for LINE in all_data_lines])
		del LINE
		train_THEN_test_list_indices_for_all_data_lines = StratifiedShuffleSplit(response_variables, n_iterations=repetitions, test_size=perc_test, random_state=rng_seed) #<*Q: Will this work for regression?><*TO DO*:[11/10/12:IN FUTURE]: CHECK>
		del response_variables
		del StratifiedShuffleSplit
	else:
		from sklearn.cross_validation import ShuffleSplit
		train_THEN_test_list_indices_for_all_data_lines = ShuffleSplit(len(all_data_lines), n_iterations=repetitions,test_size=perc_test, random_state=rng_seed)
		del ShuffleSplit
	
	train_THEN_test_list_indices_for_all_data_lines = list(train_THEN_test_list_indices_for_all_data_lines) #prior to adding this, the following assertion check failed!
	assert type([]) == type(train_THEN_test_list_indices_for_all_data_lines)
	
	train_THEN_test_list_indices_for_all_data_lines = [list(train_THEN_test_list_indices_for_all_data_lines[(rep-1)]) for rep in range(1,1+repetitions)] #prior to adding this, the following assertion check failed!
	del rep
	
	partitionedFiles = defaultdict(dict)
	
	for rep in range(1,1+repetitions):
		
		assert type([]) == type(train_THEN_test_list_indices_for_all_data_lines[(rep-1)]) 
		
		assert 0 == len(set(train_THEN_test_list_indices_for_all_data_lines[(rep-1)][0]).intersection(set(train_THEN_test_list_indices_for_all_data_lines[(rep-1)][1]))), " Train and test sets overlap for this repetition:%d !!!!" % rep
		
		####
		#12/10/12: Some indications when running generate_all_HansenAmes_model_input_files_2.py that some instances were not being included in the training or the test set! 
		#Hence the following checks were introduced!
		assert len(train_THEN_test_list_indices_for_all_data_lines[(rep-1)][0]) == len(set(train_THEN_test_list_indices_for_all_data_lines[(rep-1)][0])), " Duplicated training set indices for repetition %d ???" % rep
		assert len(train_THEN_test_list_indices_for_all_data_lines[(rep-1)][1]) == len(set(train_THEN_test_list_indices_for_all_data_lines[(rep-1)][1])), " Duplicated test set indices for repetition %d ???" % rep
		assert (len(train_THEN_test_list_indices_for_all_data_lines[(rep-1)][0])+len(train_THEN_test_list_indices_for_all_data_lines[(rep-1)][1])) == len(all_data_lines) , " Seem to have discarded some molecules by mccv repetition %d ???" % rep
		####
		
		for subset in ['TRAIN','TEST']:
			if 'TRAIN' == subset:
				indices_pos = 0
			else:
				assert 'TEST' == subset , " subset = %s ???" % subset
				indices_pos = 1
			
			line_indices = list(train_THEN_test_list_indices_for_all_data_lines[(rep-1)][indices_pos]) #prior to adding list(...), the following assertion check failed!
			assert type([]) == type(line_indices)
			line_indices.sort()
			
			output_file_name = r'%s\%s' % (output_dir,re.sub('(\.%s$)' % svmlight_file.split('.')[-1],'_mccvV%.2fR%dRng%dS%s_R%d%s.txt' % (perc_test,repetitions,rng_seed,stratified,rep,subset),svmlight_file.split("\\")[-1])) #12/10/12: Just tried to make name shorter.
			
			f_out = open(output_file_name,'w')
			try:
				for LINE_INDEX in line_indices:
					f_out.write(all_data_lines[LINE_INDEX]+'\n')
				del LINE_INDEX
			finally:
				f_out.close()
				del f_out
			
			del line_indices
			
			partitionedFiles[rep][subset] = output_file_name
	
	
	print '='*50
	
	return partitionedFiles

def mccv_partition_generic_file(dataset_file,output_dir=os.getcwd(),perc_test=0.2,repetitions=1,rng_seed=0,dataset_file_HasAHeader=False):
	#D.I.P.T.R.:<DONE>
	
	
	print '='*50
	print 'Carrying out Monte-Carlo cross-validation on %s - and writing all resultant train:test pairs to new files.' % dataset_file
	print 'Fraction of data selected for testing: ', perc_test
	print 'Number of repetitions: ', repetitions
	
	f_in = open(dataset_file)
	try:
		if dataset_file_HasAHeader:
			all_lines = [re.sub(r'\r|\n','',LINE) for LINE in f_in.readlines()]
			header = all_lines[0]
			all_data_lines = all_lines[1:]
			del all_lines
		else:
			all_data_lines = [re.sub(r'\r|\n','',LINE) for LINE in f_in.readlines()]
		del LINE
	finally:
		f_in.close()
		del f_in
	all_data_lines.sort()
	all_data_lines_indices = range(0,len(all_data_lines))
	
	import random
	random.seed(rng_seed)
	
	partitionedFiles = defaultdict(dict)
	
	for rep in range(1,1+repetitions):
		##########################
		all_data_lines_indices.sort()
		random.shuffle(all_data_lines_indices)
		test_indices = all_data_lines_indices[:int(round(len(all_data_lines_indices)*perc_test,0))]
		train_indices = [index for index in all_data_lines_indices if not index in test_indices]
		###########################
		
		assert 0 == len(set(test_indices).intersection(set(train_indices))), " For repetition %d, train and test sets will overlap???" % rep
		assert len(train_indices) == len(set(train_indices)) , " For repetition %d, there are duplicated training set indices???" % rep
		assert len(test_indices) == len(set(test_indices)) , " For repetition %d, there are duplicated training set indices???" % rep
		assert (len(train_indices)+len(test_indices)) == len(all_data_lines_indices) , " For repetition %d, we seem to be about to discard some molecules!!!!" % rep
		
		for subset in ['TRAIN','TEST']:
			if 'TRAIN' == subset:
				line_indices = train_indices
			else:
				assert 'TEST' == subset , " subset = %s ???" % subset
				line_indices = test_indices
			
			line_indices.sort()
			
			output_file_name = r'%s\%s' % (output_dir,re.sub('(\.%s$)' % dataset_file.split('.')[-1],'_mccvVal%.2fReps%dRng%d_R%d%s.txt' % (perc_test,repetitions,rng_seed,rep,subset),dataset_file.split("\\")[-1]))
			
			f_out = open(output_file_name,'w')
			try:
				if dataset_file_HasAHeader:
					f_out.write(header+'\n')
				for LINE_INDEX in line_indices:
					f_out.write(all_data_lines[LINE_INDEX]+'\n')
				del LINE_INDEX
			finally:
				f_out.close()
				del f_out
			
			del line_indices
			
			partitionedFiles[rep][subset] = output_file_name
		
	
	print '='*50
	
	return partitionedFiles
