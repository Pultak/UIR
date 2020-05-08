import math

def total_classes_usage(classes):
	res = 0
	for c in classes:
		res += classes[c][0]
	return res

def compute_word_probability(vocabulary, classes, word, classs):
	numerator = classes[classs][2][vocabulary.index(word)] + 1
	denominator = classes[classs][1] + len(vocabulary)
	return numerator / denominator

def compute_word_probability_s(vocabulary, word, classs):
	try:
		numerator  = classs[2][vocabulary.index(word)] + 1
	except ValueError:
		numerator = 1
	denominator = classs[1] + len(vocabulary)
	return numerator / denominator

def compute_class_probablity(classes, classs, file_count):
	return classes[classs][0] / file_count	

def compute_class_probablity_s(classs, total_usage):
	return classs[0] / total_usage

def classify_file(parsed_file, vocabulary, classes):
	results = {}
	prob = 1
	for c in classes:
		prob += math.log(compute_class_probablity_s(classes[c], total_classes_usage(classes)))
		for word in parsed_file:
			prob += math.log(compute_word_probability_s(vocabulary, word, classes[c]))
		results[c] = prob
		prob = 1
	return results

