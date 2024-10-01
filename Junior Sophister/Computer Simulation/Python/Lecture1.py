a, b = 3, 3.
# (a, b) = 3, 3.  # same as above

# Tuples, lists
my_tuple = (4,5,6,7)
my_list = [4,5,6,7]

mix_list = [[1,2], 'This is a string', (5, 4.2)]
# All valid inputs

# Lists are mutable, can reassign elements without creating new variables
# Tuples are immutable, cannot reassign elements

# List methods
my_list.append(8)
my_list.remove(4)
my_list.insert(0, 4)
my_list.insert(len(my_list), 9)
my_list.insert(-1, 10)
# -1 inserts in one before the last element, WEIRD
my_list = my_list[0:4]

print(my_list)

# Sets
my_set = {5, 4, 6, 7, 4}
# Elements are unique, unordered, and unindexed
my_set.add(1)
my_set.update({2, 3}) # List or set in the argument

print(my_set)

# Dictionaries
# Format is {key: value}
p_cols = {'Mercury': 'Red',
          'Venus': 'Yellow',
          'Earth': 'Blue'}
print('\nThe colours of the first three planets are: ', p_cols)
print('The colour of Earth is: ', p_cols['Earth'])

p_cols['Mars'] = 'Red' # Add a new key-value pair

# Functions
# def function_name(arg1, arg2, ...):
def my_func(x, y):
    b = y**2
    return 2*x*b, b

# Functions are used by calling them as so:
q, p = my_func(2,3)
print('\n', q, p)

# Lambda functions
# lambda arg1, arg2, ...: expression
f = lambda x, y: 2*x*y**2
print(f(2,3))

# Loops
n_loop = 10
counter = 0
while counter < n_loop:
    print('The counter value is: ', counter)
    counter += 1