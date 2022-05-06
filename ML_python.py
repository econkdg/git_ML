# ====================================================================================================
# python
# ====================================================================================================
# data type

# - int
# - float
# - bool
# - string
# - list
# ----------------------------------------------------------------------------------------------------
# variable type

from six.moves import cPickle
import matplotlib.pyplot as plt
import numpy as np

# number
x = 3
print(x)
print(type(x))

print(x + 1)  # addition
print(x - 1)  # subtraction
print(x * 2)  # multiplication
print(x / 2)  # division
print(x ** 2)  # exponentiation

x += 1
print(x)
x *= 2
print(x)

print(x/2)

x = 3.
print(x/2)

y = 2.5
print(type(y))
print(y, y + 1, y*2, y**2)

# booleans
t, f = True, False
print(type(t))  # Prints "<type 'bool'>"

print(t and f)  # Logical AND
print(t or f)  # Logical OR
print(not t)  # Logical NOT

# string
hello = 'hello'  # String literals can use single quotes
world = "world"  # or double quotes; it does not matter.
print(hello, len(hello))

hw = hello + ' ' + world  # String concatenation
print(hw)  # prints "hello world"
# ----------------------------------------------------------------------------------------------------
# container

# list
odd = [1, 3, 5, 7, 9]

a = []
b = [1, 2, 3]
c = ['I', 'love', 'deep', 'learning']
d = [1, 2, 'combination', 'of', 'string']
print(d)

d[0]
d[1]
d[2]
d[0:1]
d[0:2]
d[1:4]
d[0:0]

xs = [3, 1, 2]  # Create a list
print(xs, xs[2])
print(xs[0], xs[1])
print(xs[-1])  # Negative indices count from the end of the list; prints "2"

xs[2] = 'foo'  # Lists can contain elements of different types
print(xs)

xs.append('bar')  # Add a new element to the end of the list
print(xs)

x = xs.pop()  # Remove and return the last element of the list
print(x, xs)

# slicing
# range is a built-in function that creates a list of integers
nums = list(range(5))

print(nums)  # Prints "[0, 1, 2, 3, 4]"
print(nums[2:4])  # Get a slice from index 2 to 4 (exclusive); prints "[2, 3]"
print(nums[2:])  # Get a slice from index 2 to the end; prints "[2, 3, 4]"
# Get a slice from the start to index 2 (exclusive); prints "[0, 1]"
print(nums[:2])
print(nums[:])  # Get a slice of the whole list; prints ["0, 1, 2, 3, 4]"
print(nums[:-1])  # Slice indices can be negative; prints ["0, 1, 2, 3]"

nums[2:4] = [8, 9]  # Assign a new sublist to a slice
print(nums)  # Prints "[0, 1, 8, 8, 4]"

# condition
weekday = True
if weekday:
    print("go to the LAB")
else:
    print("wake up at 11:00 a.m and go to the LAB")

weekday = False
if weekday:
    print("go to the LAB")
else:
    print("wake up at 11:00 a.m and go to the LAB")

# loop
test_list = ['one', 'two', 'three']
for i in test_list:
    print(i)

for i in range(3):
    print(i)
    print("hi")

nums = [3, 4, 1, 5]
for num in nums:
    print(num + 1)

animals = ['cat', 'dog', 'monkey']
for animal in animals:
    print(animal)

nums = [0, 1, 2, 3, 4]  # lsit
squares = []  # empty list
for x in nums:
    squares.append(x**2)
print(squares)

nums = [0, 1, 2, 3, 4]
even_squares = [x**2 for x in nums if x % 2 == 0]
print(even_squares)

# function


def my_sum(x, y):
    return x + y


a = 3
b = 11
result = my_sum(a, b)
print(result)


def sign(x):
    if x > 0:
        return 'positive'
    elif x < 0:
        return 'negative'
    else:
        return 'zero'


for x in [-1, 0, 1]:
    print(sign(x))
# ----------------------------------------------------------------------------------------------------
# package

# import module
# import module as md
# from module import variable | function | class
# ----------------------------------------------------------------------------------------------------
# numpy

a = np.array([1, 2, 3])
print(type(a))
print(a.shape)
print(a)

a[0]
a[2]
a[0:2]

b = np.array([[1, 2, 3], [4, 5, 6]])
print(type(b))
print(b.shape)
print(b)

b[0]
b[0, 1]
b[0][1]
b[0:2, 0:2]
b[0:2, :]
# ----------------------------------------------------------------------------------------------------
# matplotlib

# example 1
x = np.arange(0, 3*np.pi, 0.1)
y = np.sin(x)

plt.plot(x, y)
plt.show()

plt.plot(x, y)
plt.axis('tight')
plt.show()

# example 2
y_sin = np.sin(x)
y_cos = np.cos(x)

plt.plot(x, y_sin)
plt.plot(x, y_cos)

plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Sine and Cosine')
plt.legend(['Sine', 'Cosine'])
plt.axis('tight')
plt.show()

# example 3
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

plt.subplot(2, 1, 1)
plt.plot(x, y_sin)
plt.axis('tight')
plt.title('Sine')


plt.subplot(2, 1, 2)
plt.plot(x, y_cos)
plt.axis('tight')
plt.title('Cosine')

plt.show()

# example 4
input_image = cPickle.load(open('./image_files/lena.pkl', 'rb'))

plt.imshow(input_image, 'gray')
plt.show()
# ====================================================================================================
