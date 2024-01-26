import numpy as np

'''
Implement the variable elimination algorithm by coding the
following functions in Python. Factors are essentially 
multi-dimensional arrays. Hence use numpy multidimensional 
arrays as your main data structure.  If you are not familiar 
with numpy, go through the following tutorial: 
https://numpy.org/doc/stable/user/quickstart.html
'''



######### restrict function
# Tip: Use slicing operations to implement this function
#
# Inputs:
# factor -- multidimensional array (one dimension per variable in the domain)
# variable -- integer indicating the variable to be restricted
# value -- integer indicating the value to be assigned to variable
#
# Output:
# resulting_factor -- multidimensional array (the dimension corresponding to variable has been restricted to value)
#########
def restrict(factor,variable,value):
    index = [slice(None)] * factor.ndim
    index[variable] = value
    resulting_factor = factor[tuple(index)]
    return resulting_factor

######### sumout function
# Tip: Use numpy.sum to implement this function
#
# Inputs:
# factor -- multidimensional array (one dimension per variable in the domain)
# variable -- integer indicating the variable to be summed out
#
# Output:
# resulting_factor -- multidimensional array (the dimension corresponding to variable has been summed out)
#########
def sumout(factor,variable):
    resulting_factor = np.sum(factor, axis=variable)
    return resulting_factor

######### multiply function
# Tip: take advantage of numpy broadcasting rules to multiply factors with different variables
# See https://numpy.org/doc/stable/user/basics.broadcasting.html
#
# Inputs:
# factor1 -- multidimensional array (one dimension per variable in the domain)
# factor2 -- multidimensional array (one dimension per variable in the domain)
#
# Output:
# resulting_factor -- multidimensional array (elementwise product of the two factors)
#########
def multiply(factor1,factor2):
    resulting_factor = factor1 * factor2
    return resulting_factor

######### normalize function
# Tip: divide by the sum of all entries to normalize the factor
#
# Inputs:
# factor -- multidimensional array (one dimension per variable in the domain)
#
# Output:
# resulting_factor -- multidimensional array (entries are normalized to sum up to 1)
#########
def normalize(factor):
    total_sum = np.sum(factor)
    resulting_factor = factor / total_sum
    return resulting_factor

######### inference function
# Tip: function that computes Pr(query_variables|evidence_list) by variable elimination.
# This function should restrict the factors in factor_list according to the
# evidence in evidence_list.  Next, it should sumout the hidden variables from the
# product of the factors in factor_list.  The variables should be summed out in the
# order given in ordered_list_of_hidden_variables.  Finally, the answer should be
# normalized to obtain a probability distribution that sums up to 1.
#
#Inputs:
#factor_list -- list of factors (multidimensional arrays) that define the joint distribution of the domain
#query_variables -- list of variables (integers) for which we need to compute the conditional distribution
#ordered_list_of_hidden_variables -- list of variables (integers) that need to be eliminated according to thir order in the list
#evidence_list -- list of assignments where each assignment consists of a variable and a value assigned to it (e.g., [[var1,val1],[var2,val2]])
#
#Output:
#answer -- multidimensional array (conditional distribution P(query_variables|evidence_list))
#########
def inference(factor_list,query_variables,ordered_list_of_hidden_variables,evidence_list):
    for variable, value in evidence_list:
        for i in range(len(factor_list)):
            if factor_list[i].shape[variable] > 1:
                factor_list[i] = restrict(factor_list[i], variable, value)
                factor_list[i] = np.expand_dims(factor_list[i], axis=variable)
                shape = list(factor_list[i].shape)
    product = factor_list[0]
    for factor in factor_list[1:]:
        product = multiply(product, factor)
    hidden_tuple = tuple(ordered_list_of_hidden_variables)
    answer = sumout(product, hidden_tuple)
    answer = normalize(answer)
    return answer

# variables
Trav = 0
FP = 1
Fraud = 2
OP = 3
Acc = 4
PT = 5
variables = np.array(['Trav', 'FP', 'Fraud', 'OP', 'Acc', 'PT'])

# values
false = 0
true = 1
values = np.array(['false', 'true'])

# P(Trav)
f1 = np.array([0.95, 0.05])
f1 = f1.reshape(2, 1, 1, 1, 1, 1)

# P(Trav, FP, Fraud)
f2 = np.array([[0.99, 0.9], [0.01, 0.1], [0.1, 0.1], [0.9, 0.9]])
f2 = f2.reshape(2, 2, 2, 1, 1, 1)

# P(Trav, Fraud)
f3 = np.array([[0.996, 0.004], [0.99, 0.01]])
f3 = f3.reshape(2, 1, 2, 1, 1, 1)

# P(Fraud, OP, Acc)
f4 = np.array([[0.9, 0.4], [0.1, 0.6], [0.7, 0.2], [0.3, 0.8]])
f4 = f4.reshape(1, 1, 2, 2, 2, 1)

# P(Acc)
f5 = np.array([0.2, 0.8])
f5 = f5.reshape(1, 1, 1, 1, 2, 1)

# P(Acc, PT)
f6 = np.array([[0.99, 0.01], [0.9, 0.1]])
f6 = f6.reshape(1, 1, 1, 1, 2, 2)

# P(Fraud)
f7 = inference([f1, f3], [Fraud], [Trav], [])
print(f"P(Fraud)={np.squeeze(f7)}\n")

# P(Fraud|FP=True, OP=False, PT=True)
f8 = inference([f1, f2, f3, f4, f5, f6], [Fraud], [Trav, Acc], [[FP, true], [OP, false], [PT, true]])
print(f"P(Fraud|FP=True, OP=False, PT=True)={np.squeeze(f8)}\n")

# P(Fraud|FP=True, OP=False, PT=True, Trav=True)
f9 = inference([f1, f2, f3, f4, f5, f6], [Fraud], [Acc], [[Trav, true], [FP, true], [OP, false], [PT, true]])
print(f"P(Fraud|FP=True, OP=False, PT=True, Trav=True)={np.squeeze(f9)}\n")

# P(Fraud|OP)
f10 = inference([f1, f2, f3, f4, f5, f6], [Fraud], [Trav, FP, Acc, PT], [[OP, true]])
print(f"P(Fraud|OP)={np.squeeze(f10)}\n")

# P(Fraud|OP, PT)
f11 = inference([f1, f2, f3, f4, f5, f6], [Fraud], [Trav, FP, Acc], [[OP, true], [PT, true]])
print(f"P(Fraud|OP, PT)={np.squeeze(f11)}\n")