
# One Reference to learn about numpy, there are many on the web
# https://www.tutorialspoint.com/numpy/numpy_pdf_version.htm
#
#
# GOAL: To start with the numpy library and to know what to do with it
#
# In this exercise you have to follow the steps and complete them with some code
# Include the Example1.py in one project in PyCharm 


print ("Using  Numpy ")

# 1. import the numpy module as np


# 2. Create an array (standard python array), called height, with this values
#  1.73, 1.68, 1.71, 1.89, 1.79


# 3. Create an array (standard python array), called weight, with this values
#  65.4, 59.2, 63.6, 88.4, 68.7


# 4. print the two arrays (height and weight) on the console



# 4. Uncomment the code and answer: What happens with this line?
#bmi = weight / height ** 2

# 5. Convert height and weight into two numpy arrays,
#  called np_height and np_weight, respectively


# 6. Compute bmi again with the two numpy arrays
# bmi is the weight / height^2



# 7. print bmi array on the console

# 8. Compute the mean of the bmi and print on the console

# 9. Compute the median  of the bmi and print on the console

# 10. Compute the correlation between height and weight

# 11. Compute the standard deviation of bmi

# 12. Show on console the dimension of bmi, height and weight

# 13a. Compute the sum of bmi and sort the values too, show bmi on the console


# 13b. The result for these sentences is the same?
# Think about it before uncomment the code and run it to look at the results
#print(height + weight)
#print (np_height + np_weight)

# 14. Create a 2 dimensional array, called np_table,
# in one dimension the values of the height and
# in the second dimension the values of the weight



# 15. Print on the console the first dimension of np_table

# 16. Print on the console the second dimension of np_table


# 17. Print on the console all the dimension of the np_table


# 18. Print on the console the second column of np_table


# 19. Print on the console the third column of np_table


#  20 Create and print an array, called np_tableshort,  with the second and third columns of np_table


# end of the  numpy code


# OUTPUT OF THE CODE IN CONSOLE 

#Using  Numpy 
#[1.73, 1.68, 1.71, 1.89, 1.79]
#[65.4, 59.2, 63.6, 88.4, 68.7]
#[21.85171573 20.97505669 21.75028214 24.7473475  21.44127836]
#22.15313608308522
#21.750282138093777
#[[1.         0.97514969]
# [0.97514969 1.        ]]
#1.3324932175111337
#(5,)
#110.76568041542609
#[20.97505669 21.44127836 21.75028214 21.85171573 24.7473475 ]
#[[ 1.73  1.68  1.71  1.89  1.79]
# [65.4  59.2  63.6  88.4  68.7 ]]
#(2, 5)
#[1.73 1.68 1.71 1.89 1.79]
#[65.4 59.2 63.6 88.4 68.7]
#[ 1.71 63.6 ]
#[[ 1.68  1.71]
# [59.2  63.6 ]]
#
#
#
# Additionally, follow the suggestion
# It is interesting you also learn how to use matplotlib
# there are a lot of tutorials on the web, below you have just one example
# https://matplotlib.org/Matplotlib.pdf
# Additionally, you have a few examples, just run to know that
# you have installed it correctly at your computer/laptop

# uncomment the code and run it

# import matplotlib.pyplot as plt
#
#
# x = np.linspace(0, 2, 100)
# plt.plot(x, x, label='linear')
# plt.plot(x, x**2, label='quadratic')
# plt.plot(x, x**3, label='cubic')
# plt.xlabel('x label')
# plt.ylabel('y label')
# plt.title("Simple Plot")
# plt.legend()
# plt.show()
#
# x = np.arange(0, 10, 0.2)
# y = np.sin(x)
# fig, ax = plt.subplots()
# ax.plot(x, y)
# plt.show()
#
#
# names = ['group_a', 'group_b', 'group_c']
# values = [1, 10, 100]
# plt.figure(1, figsize=(9, 3))
# plt.subplot(131)
# plt.bar(names, values)
# plt.subplot(132)
# plt.scatter(names, values)
# plt.subplot(133)
# plt.plot(names, values)
# plt.suptitle('Categorical Plotting')
# plt.show()
