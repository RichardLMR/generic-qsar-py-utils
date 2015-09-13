#########################################################################################################
# test_13.py
# Implements unit tests for the genericQSARpyUtils project (see below).
#
# ########################################
# #test_13.py: Key documentation :Contents#
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
current_dir = r'%s' % os.path.dirname(os.path.abspath(__file__))
project_modules_to_test_dir = os.path.split(os.path.split(current_dir)[0])[0]
sys.path.append(project_modules_to_test_dir)

import unittest
sys.path.append(r'%s\tests\test_10' % project_modules_to_test_dir)
from test_10 import test_10

class test_13(test_10):
	
	def test_13_filter_features_for_svmlight_format_files_BinClassExample(self):
		##############################
		print 'Running unittests for this project: ', project_name
		print 'Running this unittest: ', self._testMethodName
		##################################
		
		do_not_to_delete = glob.glob(r'%s\*' % current_dir)
		
		from ml_functions import filter_features_for_svmlight_format_files
		
		###########
		#c.f. generate_modelling_input.py:
		
		from ml_functions import chi2 as fsFunction
		
		subset2fsFile,indices_of_top_K_features = filter_features_for_svmlight_format_files(svmlight_format_train_file=r'%s\toy13_complete_svmlight_file_2-FoldCVR3SFalse_R1F1TRAIN.txt' % current_dir,svmlight_format_test_file=r'%s\toy13_complete_svmlight_file_2-FoldCVR3SFalse_R1F1TEST.txt' % current_dir,univariate_scoring_function=fsFunction,number_of_features_to_retain=1,ensure_test_set_consistency=True)
		
		del indices_of_top_K_features #not currently interested
		###########
		
		f_out = open(r'%s\test_13_output_names.txt' % current_dir,'wb')
		try:
			for subset in ['TRAIN','TEST']:
				f_out.write('%s=%s\r\n' % (subset,os.path.relpath(subset2fsFile[subset],start=current_dir)))
		finally:
			f_out.close()
			del f_out
		
		self.compareAllExpectedAndActualFiles(current_dir)
		
		self.clean_up_if_all_checks_passed(current_dir,specific_files_not_to_delete=do_not_to_delete)
