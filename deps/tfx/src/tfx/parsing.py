
# Copyright 2015 Ben Kehoe
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import math, numbers
from math import *
import collections

__version__ = '0.1'

def perform_replacements(string,replacements={},keyword_callback=None,surround_with_parens=False,pi=False,deg=False):

	for rep in replacements.items():
		key = rep[0]
		value = str(rep[1]).strip()
		if surround_with_parens:
			value = '(' + value + ')'
		if not re.search(r'\A[a-zA-z]\w+\Z',key):
			raise Exception('Replacement name %s is invalid' % key)
		string = re.sub(r'\$\(\s*' + key + r'\s*\)',value,string)
	
	res = re.search(r'\$\(\s*([a-zA-z]\w+)\s*\)',string)
	if res:
		raise Exception('No value given for replacement %s!' % res.group(1))

	res = re.search(r'\$\(\s*([a-zA-Z]+)\s+([a-zA-Z]\w*)\s*\)',string)
	if res and not keyword_callback:
		raise Exception('no callback')
	while res:
		name = res.group(1)
		arg = res.group(2)
		if isinstance(keyword_callback,dict):
			print keyword_callback[name]
			value = str(keyword_callback[name](arg))
		else:
			value = str(keyword_callback(name,arg))
		if surround_with_parens:
			value = '(' + value + ')'
		string = re.sub(r'\$\(\s*'+name+'\s+'+arg+'\s*\)',value,string,count=1)
		res = re.search(r'\$\(\s*([a-zA-Z]+)\s+([a-zA-Z]\w*)\s*\)',string)

	res = re.search(r'\$\(([^\)]*)\)',string)
	if res:
		raise Exception('Invalid replacement name %s' % res.group(1))
	
	if pi:
		pi_str = '%.16f' % math.pi
		if surround_with_parens:
			pi_str = '(' + pi_str + ')'
		string = re.sub(r'(?<=\W)[pP][iI](?!\w)',pi_str,string)
		string = re.sub(r'(?<=\A)[pP][iI](?!\w)',pi_str,string)
	
	if deg:
		deg_str = '*(%.16f/180)' % math.pi
		string = re.sub(r'(?<=\W|\d)deg(?!\w)',deg_str,string)
	
	return string

def parse_number(val,replacements={},keyword_callback=None):
	if isinstance(val,numbers.Number):
		return val
	string = str(val)
	string = perform_replacements(string,replacements,keyword_callback,surround_with_parens=True,pi=True,deg=True)
	
	#remove spaces
	string = re.sub(r'\s+','',string.lower())

	string = re.sub(r'\-\+','-',string)
	string = re.sub(r'\+\-','-',string)
	string = re.sub(r'\+\+','+',string)

	#numbers -> N
	Nstring = re.sub(r'\d+(\.\d*)?|\.\d+','N',string)

	whitelist = ('sin','cos','tan','asin','acos','atan','fabs','sqrt')

	while True:
		Nstr_at_start = Nstring

		print Nstring
		
		#simple binary op
		# N[+-*/]N -> N
		while re.search(r'N[\+\-\*\/]N',Nstring):
			Nstring = re.sub(r'N[\+\-\*\/]N','N',Nstring)
		print Nstring

		#single num in parens is just num
		# (([+-]N)) -> (N)
		while re.search(r'\(\([\+\-]?N\)\)',Nstring):
			Nstring = re.sub(r'\(\([\+\-]?N\)\)',r'(N)',Nstring)

		print Nstring
		
		# [+-*/](N) -> [+-*/]N
		while re.search(r'([\+\-\*\/])\(N\)',Nstring):
			Nstring = re.sub(r'([\+\-\*\/])\(N\)',r'\1N',Nstring)
		print Nstring

		# ^(N)[+-*/]-> N[+-*/]
		Nstring = re.sub(r'\A\(N\)([\+\-\*\/])',r'N\1',Nstring)
		print Nstring

		#initial pos/neg sign
		# ^[+-]N -> N
		Nstring = re.sub(r'\A[\+\-]N',r'N',Nstring)
		print Nstring

		# check for [+-*/]sin(N), etc., replace with N
		while re.search(r'(?<!\w)(' + '|'.join(whitelist) + r')\(N\)',Nstring):
			Nstring = re.sub(r'(?<!\w)(' + '|'.join(whitelist) + r')\(N\)','N',Nstring)

		print Nstring
		if Nstr_at_start == Nstring:
			if Nstring != 'N':
				raise Exception('did not work!')
			break
	
	try:
		num = eval(string)
		return num
	except Exception, e:
		print e
		raise Exception('Failed')
		

def parse_number_sequence(val,expected_len=None,replacements={},keyword_callback=None,strict_separation=False):
	if type(val) is not str and isinstance(val,collections.Sequence):
		return [parse_number(field) for field in val]
	
	string = re.sub(r' +',' ',re.sub(r'[\[\]]',' ',str(val))).strip()
	string = re.sub(r'\s*[,;\t]\s*',',',string)

	string = re.sub(r'\s*\-\s*\+\s*','-',string)
	string = re.sub(r'\s*\+\s*\-\s*','-',string)
	string = re.sub(r'\s*\+\s*\+\s*','+',string)

	string = perform_replacements(string,replacements,keyword_callback,surround_with_parens=True,pi=True,deg=True)
	
	cnt = string.count(',')+1
	if cnt != 1:
		if expected_len == cnt-1 and string[-1] == ',':
			string = string[:-1]
		elif expected_len and cnt != expected_len:
			raise Exception('Delimited into %d fields, but expected %d' % (cnt,expected_len))

		fields = string.split(',')
		return [parse_number(field) for field in fields]
	
	#find possible breaks, and also check parentheses nesting
	print string
	nest_level=0
	possible_spaces = []
	for i in xrange(len(string)):
		if string[i] == '(':
			nest_level += 1
		elif string[i] == ')':
			nest_level -= 1
		
		if nest_level < 0: raise Exception
		
		if nest_level == 0 and string[i] == ' ' \
				and re.search(r'[^\+\-\*\/]\Z',string[:i]) \
				and re.search(r'\A[^\*\/]',string[i+1:]):
			possible_spaces.append(i)

	if nest_level != 0: raise Exception

	# check num possible spaces
	possible_possibilities = [possible_spaces,
			[ps for ps in possible_spaces if re.search(r'\A[^\+]',string[ps+1:])],
			[ps for ps in possible_spaces if re.search(r'\A[^\+\-]',string[ps+1:])]]
	
	if strict_separation:
		del possible_possibilities[1:]
	elif expected_len is None and not \
			(possible_possibilities[0] == possible_possibilities[1] and \
			possible_possibilities[1] == possible_possibilities[2]):
		raise Exception('Ambiguous space-separated values')
	print possible_possibilities

	fields = []
	for p in possible_possibilities:
		if expected_len is None or len(p)+1 == expected_len:
			splitvals = [-1] + p
			for i in range(len(p)+1):
				if i+1 <= len(p):
					fields.append(string[splitvals[i]+1:splitvals[i+1]])
				else:
					fields.append(string[splitvals[i]+1:])
			return [parse_number(field) for field in fields]
	else:
		raise Exception('Couldn\'t find spaces to parse')
