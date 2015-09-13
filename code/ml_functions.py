#########################################################################################################
# ml_functions.py
# One of the Python modules written as part of the genericQSARpyUtils project (see below).
#
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
import numpy as np
import re,os
from collections import defaultdict
import functools
import sklearn

def chi2(X,Y):
	return sklearn.feature_selection.chi2(X,Y)

def f_regression(X,Y):
	return sklearn.feature_selection.f_regression(X,Y,center=False)

def renameFilteredFeaturesFile(original_file,name_of_feature_selection_method,number_of_features_retained,ensure_test_set_consistency):
	file_format = original_file.split('.')[-1]
	
	if not ensure_test_set_consistency:
		new_file_name = re.sub('(\.%s$)' % file_format,'_fs_%s_top_%d.%s' % (name_of_feature_selection_method,number_of_features_retained,file_format),original_file)
	else:
		new_file_name = re.sub('(\.%s$)' % file_format,'_nonDefault_fs_%s_top_%d.%s' % (name_of_feature_selection_method,number_of_features_retained,file_format),original_file)
		
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
	try:
		del indexValuePAIR #This gave the following error (1) [hence now catching] when processing the following file (2): (1) "UnboundLocalError: local variable 'indexValuePAIR' referenced before assignment", (2) C:\Work\rffc\benchmarks\datasets\MUV\MUV_aid712_svmlightCMS_5-FoldCVR2STrue_R1F1TRAIN.txt
	except:
		pass
	
	new_LINE = '%s ' % class_label+' '.join(indexValuePairsToRetain)+'#%s' % ID
	
	return new_LINE

def update_svmlight_line_to_ensure_test_set_consistency(LINE_WITHOUT_LINE_ENDING,indices_of_features_to_retain):
	
	########################################################
	#Required if then convert svmlight file into csv format#
	#Only need to apply this update to a single line to ensure all retained features are present in both the training and test set.
	########################################################
	
	class_then_descriptors_info, ID = LINE_WITHOUT_LINE_ENDING.split('#') 
	
	class_label = class_then_descriptors_info.split()[0]
	indexValuePairs = class_then_descriptors_info.split()[1:]
	
	del class_then_descriptors_info
	del LINE_WITHOUT_LINE_ENDING
	
	indexValuePairsToRetain = [indexValuePAIR for indexValuePAIR in indexValuePairs if int(indexValuePAIR.split(':')[0]) in indices_of_features_to_retain]
	try:
		del indexValuePAIR
	except:
		pass
	
	indexValuePairsToRetain += ['%d:0' % missing_feature for missing_feature in list(set(indices_of_features_to_retain).difference(set([int(pair.split(':')[0]) for pair in indexValuePairs])))]
	
	try:
		del pair
		del missing_feature
	except:
		pass
	
	new_LINE = '%s ' % class_label+' '.join(indexValuePairsToRetain)+'#%s' % ID
	
	return new_LINE

def remove_extra_features(orig_svmlight_format_file,new_svmlight_format_file,indices_of_features_to_retain,ensure_test_set_consistency):
	
	f_in = open(orig_svmlight_format_file)
	try:
		orig_data = [re.sub(r'\r|\n','',LINE) for LINE in f_in.readlines()]
		del LINE
	finally:
		f_in.close()
		del f_in
		
	
	f_out = open(new_svmlight_format_file,'w')
	try:
		
		line_count = 0
		for ORIG_LINE in orig_data:
			line_count += 1
			
			if 1 == line_count and ensure_test_set_consistency:
				f_out.write(update_svmlight_line_to_ensure_test_set_consistency(filter_svmlight_format_line(ORIG_LINE,indices_of_features_to_retain),indices_of_features_to_retain)+'\n')
			else:
				f_out.write(filter_svmlight_format_line(ORIG_LINE,indices_of_features_to_retain)+'\n')
	finally:
		f_out.close()
		del f_out

def filter_features_for_svmlight_format_files(svmlight_format_train_file,svmlight_format_test_file=None,univariate_scoring_function=chi2,number_of_features_to_retain=200,ensure_test_set_consistency=False):
	####################################################
	#d.i.p.t.r.(including called functions): <TO DO>####
	####################################################
	'''
	This function can write out svmlight format versions of the training set *and*, if a corresponding test set file is specified, the test set, based only on the features selected using information from the training set.
	<*N.B.*: Only certain unvariate_scoring_function values will give sensible results for classification/regression datasets!
	-e.g. chi2 is appropriate for classification datasets.>
	If ensure_test_set_consistency = True (not default), then zero-valued feature entries will be created for selected features in the test set where this feature did not actually appear in the test set pre-filter file.
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
	
	assert X_train_data.shape[1] >= number_of_features_to_retain , " You cannot ask for %d features to be selected if the *training set* only contains %d!" % (number_of_features_to_retain,X_train_data.shape[1])
	
	indices_of_top_K_features = report_top_K_features(X_train_data,Y_train_data,univariate_scoring_function,number_of_features_to_retain)
	
	
	print '='*50
	print 'Selected the features (i.e. descriptors) corresponding to these indices: '
	print indices_of_top_K_features
	print '='*50
	
	assert len(indices_of_top_K_features) == number_of_features_to_retain, " You asked for %d features to be retained. You got %d???" % (len(indices_of_top_K_features),number_of_features_to_retain)
	
	#######################
	#Writing output files##
	#######################
	
	output_files = {} #N.B.: Added after original test of this function [but no other changes made].
	
	for TRAIN_OR_TEST_LABEL in ['TRAIN','TEST']:
		if 'TRAIN' == TRAIN_OR_TEST_LABEL:
			orig_svmlight_format_file = svmlight_format_train_file
		else:
			assert 'TEST' == TRAIN_OR_TEST_LABEL
			orig_svmlight_format_file = svmlight_format_test_file
		
		if orig_svmlight_format_file is None:
			assert 'TEST' == TRAIN_OR_TEST_LABEL , " orig_svmlight_format_file (for the training set) is deemed to be None???"
			continue
		
		new_svmlight_format_file = renameFilteredFeaturesFile(original_file=orig_svmlight_format_file,name_of_feature_selection_method=univariate_scoring_function.__name__,number_of_features_retained=number_of_features_to_retain,ensure_test_set_consistency=ensure_test_set_consistency)
		
		output_files[TRAIN_OR_TEST_LABEL] = new_svmlight_format_file
		
		remove_extra_features(orig_svmlight_format_file,new_svmlight_format_file,indices_of_top_K_features,ensure_test_set_consistency)
	
	#######################
	
	return output_files, indices_of_top_K_features #It may be interesting to analyse those features which were consistently selected across multiple train:test partitions in future work!


def crossValidate_svmlight_file(svmlight_file,output_dir=os.getcwd(),mccv=True,folds=1,perc_test=0.2,repetitions=1,stratified=True,only_write_IDs=False):
	#0.1 -> 0.2: Replaces mccv_svmlight_file(...)
	import random
	##########################################
	#d.i.a.test.run.fin.ok(inc.args.):<DONE>-BUT WHILST DOING SO, MADE A FEW CHANGES; ALTHO' I THINK I D.I. THESE ADEQUATELY AS WELL => D.I. AGAIN?<TO DO>?##
	##########################################
	
	####################################################################################
	#<DONE for binary classification/regression dataset => test_5/test_6 - updated this note on 02/06/13 without updating code!>: WRITE TESTS FOR THIS FUNCTION##
	####################################################################################
	
	print '='*50
	print 'Carrying out cross-validation on %s - and writing all resultant train:test pairs to new files.' % svmlight_file
	if only_write_IDs:
		print 'Will only write out instance IDs - such that train/test pairs can be generated consistently using the predefined_split_svmlight_file(...) function!'
	if mccv:
		print 'Type of cross-validation: Monte-Carlo cross-validation.'
		assert 1 == folds , "Only one fold is allowed for Monte-Carlo cross-validation!"
		print 'Fraction of data selected for testing: ', perc_test
		assert (type(0.5) == type(perc_test) and perc_test > 0 and perc_test < 1)
	else:
		print 'Type of cross-validation: %d-fold CV.' % folds
	print 'Number of repetitions: ', repetitions
	print 'Stratified: %s.' % stratified
	print '='*50
	
	
	f_in = open(svmlight_file)
	try:
		all_data_lines = [re.sub(r'\r|\n','',LINE) for LINE in f_in.readlines()]
		del LINE
	finally:
		f_in.close()
		del f_in
	
	if mccv:
		#16/03/12: see: http://scikit-learn.org/0.11/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html#sklearn.cross_validation.StratifiedShuffleSplit ; http://scikit-learn.org/0.11/modules/generated/sklearn.cross_validation.ShuffleSplit.html#sklearn.cross_validation.ShuffleSplit
		if stratified:
			response_variables = np.array([float(LINE.split()[0]) for LINE in all_data_lines])
			del LINE
			#################################################
			#<16/03/13: Check for failure expected later on>#
			#################################################
			if 0 == len([Y for Y in list(response_variables) if not int(Y)==Y]): 
				del Y
				print 'Dealing with discrete numbers in response_variables => classification data!'
				classes = list(set(list(response_variables)))
				class2size = dict(zip([CLASS for CLASS in classes],[len([Y for Y in list(response_variables) if Y == CLASS]) for CLASS in classes]))
				from operator import itemgetter
				size_of_smallest_class = sorted(class2size.iteritems(), key=itemgetter(1))[0][1]
				
				assert int(perc_test*size_of_smallest_class) == perc_test*size_of_smallest_class , " perc_test = %f will not work with mccv=True and stratified=True for your data, since perc_test*size_of_smallest_class(where smallest class = %d and size_of_smallest_class=%d) is not an integer!" % (perc_test,sorted(class2size.iteritems(), key=itemgetter(1))[0][0],size_of_smallest_class)
				del classes
				del class2size
				del size_of_smallest_class
				del itemgetter
				
			del Y
			#################################################
			
			
			from sklearn.cross_validation import StratifiedShuffleSplit
			train_THEN_test_list_indices_for_all_reps = StratifiedShuffleSplit(response_variables, n_iterations=repetitions, test_size=perc_test,random_state=0) #<*Q: Will this work for regression?><*TO DO*:[11/10/12:IN FUTURE]: CHECK>
			del response_variables
			del StratifiedShuffleSplit
		else:
			from sklearn.cross_validation import ShuffleSplit
			train_THEN_test_list_indices_for_all_reps = ShuffleSplit(len(all_data_lines), n_iterations=repetitions,test_size=perc_test,random_state=0) #<*Q: Will this work for regression?><*TO DO*:[11/10/12:IN FUTURE]: CHECK>
			del ShuffleSplit
		
		train_THEN_test_list_indices_for_all_reps = list(train_THEN_test_list_indices_for_all_reps) #prior to adding this, the following assertion check failed!
		assert type([]) == type(train_THEN_test_list_indices_for_all_reps)
		
		train_THEN_test_list_indices_for_all_reps = [list(train_THEN_test_list_indices_for_all_reps[(rep-1)]) for rep in range(1,1+repetitions)] 
		del rep
		train_THEN_test_list_indices_for_all_folds_for_all_reps = [[[list(train_THEN_test_pair[0]),list(train_THEN_test_pair[1])]] for train_THEN_test_pair in train_THEN_test_list_indices_for_all_reps]
		del train_THEN_test_list_indices_for_all_reps
		
	else:
		#16/03/12: see:http://scikit-learn.org/0.11/modules/generated/sklearn.cross_validation.StratifiedKFold.html#sklearn.cross_validation.StratifiedKFold ; http://scikit-learn.org/0.11/modules/generated/sklearn.cross_validation.KFold.html#sklearn.cross_validation.KFold
		#####################################################################################################################
		#<N.B.!> 16/03/2013: also see: http://stackoverflow.com/questions/8281034/1010-fold-cross-validation-in-scikit-learn#
		#=> KFold(...) and StratifiedKFold(...) will give the same indices!!! => It would be best to make different repetitions give independent, random partitions by shuffling the data after generating the indices once!
		#####################################################################################################################
		
		train_THEN_test_list_indices_for_all_folds_for_all_reps = []
		
		every_rep_all_folds_train_THEN_test_indices = []
		
		if stratified:
			from sklearn.cross_validation import StratifiedKFold
			response_var = np.array([float(LINE.split()[0]) for LINE in all_data_lines])
			kf = StratifiedKFold(response_var,folds)#,random_state=0)
			del StratifiedKFold
			del response_var
		else:
			from sklearn.cross_validation import KFold
			kf = KFold(len(all_data_lines),folds)#,random_state=0)
			del KFold
		for train,test in kf:
			every_rep_all_folds_train_THEN_test_indices.append([list(train),list(test)])
		del kf
		
		assert len(every_rep_all_folds_train_THEN_test_indices) == folds
		
		###############################################
		######<Tests which are specific to k-fold cv>##
		###############################################
		all_test_indices = []
		for FOLD in range(0,folds):
			all_test_indices += every_rep_all_folds_train_THEN_test_indices[FOLD][1]
		del FOLD
		assert len(all_data_lines) == len(all_test_indices)
		assert len(all_test_indices) == len(set(all_test_indices))
		del all_test_indices
		################################################
		
		for rep in range(0,repetitions):
			train_THEN_test_list_indices_for_all_folds_for_all_reps.append(every_rep_all_folds_train_THEN_test_indices)
		del rep
		del every_rep_all_folds_train_THEN_test_indices
	
	
	
	assert len(train_THEN_test_list_indices_for_all_folds_for_all_reps) == repetitions
	assert 0 == len([LIST for LIST in train_THEN_test_list_indices_for_all_folds_for_all_reps if not len(LIST) == folds])
	del LIST
	
	if only_write_IDs:
		all_data_lines = [LINE.split('#')[1] for LINE in all_data_lines] #Now, all_data_lines should actually just be a list of IDs!
		del LINE
	
	partitionedFiles = defaultdict(functools.partial(defaultdict,dict))
	for rep in range(1,1+repetitions):
		random.seed(rep)
		random.shuffle(all_data_lines) #to make sure StratifiedKFold(...) and KFold(...) do not give the same folds with each repetition even tho' the line indices generated above will be the same for each repetition!
		
		for FOLD in range(1,1+folds):
			assert type([]) == type(train_THEN_test_list_indices_for_all_folds_for_all_reps[(rep-1)][(FOLD-1)]) 
			
			assert 0 == len(set(train_THEN_test_list_indices_for_all_folds_for_all_reps[(rep-1)][(FOLD-1)][0]).intersection(set(train_THEN_test_list_indices_for_all_folds_for_all_reps[(rep-1)][(FOLD-1)][1]))), " Train and test sets overlap for this repetition:%d and this fold:%d !!!!" % (rep,FOLD)
			
			####
			#12/10/12: Some indications when running generate_all_HansenAmes_model_input_files_2.py that some instances were not being included in the training or the test set! 
			#Hence the following checks were introduced!
			assert len(train_THEN_test_list_indices_for_all_folds_for_all_reps[(rep-1)][(FOLD-1)][0]) == len(set(train_THEN_test_list_indices_for_all_folds_for_all_reps[(rep-1)][(FOLD-1)][0])), " Duplicated training set indices for repetition %d and fold %d???" % (rep,FOLD)
			assert len(train_THEN_test_list_indices_for_all_folds_for_all_reps[(rep-1)][(FOLD-1)][1]) == len(set(train_THEN_test_list_indices_for_all_folds_for_all_reps[(rep-1)][(FOLD-1)][1])), " Duplicated test set indices for repetition %d and fold %d???" % (rep,FOLD)
			combined_indices = train_THEN_test_list_indices_for_all_folds_for_all_reps[(rep-1)][(FOLD-1)][0]+train_THEN_test_list_indices_for_all_folds_for_all_reps[(rep-1)][(FOLD-1)][1]
			combined_indices.sort()
			assert combined_indices == range(0,len(all_data_lines)) , " Seem to have discarded some molecules for repetition %d  and fold %d???" % (rep,FOLD) #16/03/12: This seems to happen when mccv=True, stratified=True and (perc_test x [number of instances in the smallest class]) is not an integer! => Added a check for this at the start of this function!
			####
			
			for subset in ['TRAIN','TEST']:
				if 'TRAIN' == subset:
					indices_pos = 0
				else:
					assert 'TEST' == subset , " subset = %s ???" % subset
					indices_pos = 1
				
				line_indices = list(train_THEN_test_list_indices_for_all_folds_for_all_reps[(rep-1)][(FOLD-1)][indices_pos]) #prior to adding list(...), the following assertion check failed!
				assert type([]) == type(line_indices)
				line_indices.sort()
				
				if mccv:
					output_file_name = r'%s\%s' % (output_dir,re.sub('(\.%s$)' % svmlight_file.split('.')[-1],'_mccvV%.2fR%dS%s_R%d%s.txt' % (perc_test,repetitions,stratified,rep,subset),svmlight_file.split("\\")[-1])) #12/10/12: Just tried to make name shorter.
				else:
					output_file_name = r'%s\%s' % (output_dir,re.sub('(\.%s$)' % svmlight_file.split('.')[-1],'_%d-FoldCVR%dS%s_R%dF%d%s.txt' % (folds,repetitions,stratified,rep,FOLD,subset),svmlight_file.split("\\")[-1]))
				
				if only_write_IDs:
					output_file_name = re.sub('(\.txt$)','_onlyIDs.txt',output_file_name)
				
				f_out = open(output_file_name,'w')
				try:
					for LINE_INDEX in line_indices:
						f_out.write(all_data_lines[LINE_INDEX]+'\n')
					del LINE_INDEX
				finally:
					f_out.close()
					del f_out
				
				del line_indices
				
				partitionedFiles[rep][FOLD][subset] = output_file_name
				
		del FOLD
	del rep
	
	
	print '='*50
	
	return partitionedFiles


def predefined_split_svmlight_file(svmlight_file,train_IDs_file,test_IDs_file,output_label=''):
	#<DONE=><OK>>: d.i.p.t.f.r
	print '='*50
	print 'Partitioning %s into a training and test set based upon \n the train IDs read from %s \n and the test IDs read from %s \n Labelling train:test pair of files using the following label: \n %s \n' % (svmlight_file,train_IDs_file,test_IDs_file,output_label)
	
	
	
	subsetIDsDict = defaultdict(list)
	for subset in ['TRAIN','TEST']:
		if 'TRAIN' == subset:
			IDs_file = train_IDs_file
		else:
			assert 'TEST' == subset
			IDs_file = test_IDs_file
		f_in = open(IDs_file)
		try:
			subsetIDsDict[subset] = [re.sub(r'\r|\n','',LINE) for LINE in f_in.readlines()]
			del LINE
		finally:
			f_in.close()
			del f_in
	del subset
	
	assert 0 == len(set(subsetIDsDict['TRAIN']).intersection(set(subsetIDsDict['TEST']))) , " Train and test IDs overlap???"
	assert len(subsetIDsDict['TRAIN']+subsetIDsDict['TEST']) == len(set(subsetIDsDict['TRAIN']+subsetIDsDict['TEST'])) , " Duplicate IDs in train/test set???"
	
	f_in = open(svmlight_file)
	try:
		all_lines = [re.sub(r'\r|\n','',LINE) for LINE in f_in.readlines()]
		del LINE
	finally:
		f_in.close()
		del f_in
	
	id2Line = dict(zip([LINE.split('#')[1] for LINE in all_lines],[LINE for LINE in all_lines]))
	del LINE
	
	assert len(id2Line) == len(all_lines) , " Duplicate IDs in complete dataset file???"
	all_ids = id2Line.keys()
	all_ids.sort()
	train_plus_test_ids = subsetIDsDict['TRAIN']+subsetIDsDict['TEST']
	train_plus_test_ids.sort()
	assert not 0 == len(all_ids)
	assert all_ids == train_plus_test_ids
	del all_ids
	del train_plus_test_ids
	
	subset2svmlightFile = {}
	
	for subset in subsetIDsDict:
		file = re.sub('(\.%s$)' % svmlight_file.split('.')[-1], '%s%s.txt' % (output_label,subset),svmlight_file)
		f_out = open(file,'w')
		try:
			for ID in subsetIDsDict[subset]:
				f_out.write(id2Line[ID]+'\n')
		finally:
			f_out.close()
			del f_out
		subset2svmlightFile[subset] = file
	
	print '='*50
	
	return subset2svmlightFile
