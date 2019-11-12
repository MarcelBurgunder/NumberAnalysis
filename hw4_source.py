# python libraries
import os

# numpy libraries
import numpy as np

# matplotlib libraries
import matplotlib.pyplot as plt

# math libraries
import math

#globals
maxn2 = 15
a_2 = -5
b_2 = 5
interval_size_2 = b_2-a_2
Nodes = np.zeros(shape=(6, 3, maxn2+1)) # Nodes for question 2 --> 5a, 10a, 15a, 5b, 10b, 15b; then na/b, x, f(x)
a_3 = 0
b_3 = 1
interval_size_3 = b_3-a_3
points = 1000	# for both question 2 and 3
n3 = 20	# the "n" computed for question 3, part a; n = 20 was computed, but experimentally with this model
		# n = 7 yields an error always less than 10^-2 --> change this value for question 3 for any n
A = np.zeros(shape=(n3,2))	# spline constants, dimensions represent: gradient(m), offset(c) for y=mx+c

def abs(x):
	if (x < 0):
		return x*(-1)
	else:
		return x

def f2(x):	# function for question 2; f(x) = 1/(1+x^2)
	return 1/(1+(x**2))

def pn(x, ni):	# polynomial of function for question 2; input x, and global Nodes index ni

	sum = 0			# Lagrange Polynomials
	number_of_nodes = int(Nodes[ni][0][0] + 1)
	for i in range(number_of_nodes):
		product = Nodes[ni][2][i]	# starts as f(xk)
		xk = Nodes[ni][1][i]		# denotes xk
		for j in range(number_of_nodes):
			if (xk == Nodes[ni][1][j]):
				continue
			product *= (x-Nodes[ni][1][j])
			product /= (xk-Nodes[ni][1][j])
		sum += product
	return sum

def en(x, ni):	# x input, and ni for pn
	return f2(x)-pn(x, ni)

def f3(x):	# function of x for question 3 e^(-2x)+2x^2+x+1
	return math.exp(-2*x)+2*(x**2)+x+1

def S(n, x):	# finds the correct spline, takes constant values for correct spline
	dx = interval_size_3/n
	si = int(x/dx)
	return (A[si][0]*x)+A[si][1]

def absdif_f_S(n, x, cx=1):	# absolute difference between f and S; use formula (x-x0)(x-x1)...(x-xn)/(n+1)!f(n)(x)

	product = x
	for i in range(1, n):
		product *= ((x-(i/n))/i)
	fn = (-2**n)*math.exp(cx)

	if (n > 2):
		return abs(product*(fn))
	if (n == 2):
		return abs(product*(fn + 4))
	if (n == 1):
		return abs(product*(fn + (4*cx) + 1))
	if (n == 0):
		return abs(product*(fn + 2*(cx**2) + cx + 1))

def main():

	'''
	Question 2
	'''

	# First fill in the global array of nodes

	for i in range(3):
		n2 = 5 + (i*5)
		Nodes[i][0][0] = n2
		Nodes[i+3][0][0] = n2
		for j in range(n2+1):
			xa = a_2 + (j*(interval_size_2/n2))			# equally-spaced nodes
			p = math.cos((math.pi/2)*(((2*j)+1)/n2))	# Chebyshev nodes
			xb = ((a_2+b_2)/2)+(((b_2-a_2)/2)*p)
			Nodes[i][1][j] = xa
			Nodes[i][2][j] = f2(xa)
			Nodes[i+3][1][j] = xb
			Nodes[i+3][2][j] = f2(xb)

			# First dimension denotes the n value and question number
			# Second dimension denotes whether n indicator, x or f(x)
			# Third dimension are different nodes

	'''
	Part a
	'''

	q2a = np.zeros(shape=(3, 5, points+1))	# Question 2a, dimensions: n-value; input type; data point
	for i in range(3):
		n = 5 + (i*5)
		for j in range(points+1):
			q2a[i][0][j] = n
			x = a_2+(j*(interval_size_2/points))
			q2a[i][1][j] = x
			q2a[i][2][j] = f2(x)
			q2a[i][3][j] = pn(x, i)
			q2a[i][4][j] = en(x, i)

	for i in range(3):
		plt.plot(q2a[i,1,:], q2a[i,2,:])
		plt.plot(q2a[i,1,:], q2a[i,3,:])
		plt.xlabel("x value")
		plt.title("Plot for n = " + str(q2a[i][0][0]) + "; with equally spaced nodes")
		plt.show()

		plt.plot(q2a[i,1,:], q2a[i,4,:])
		plt.xlabel("x value")
		plt.ylabel("en")
		plt.title("Plot for n = " + str(q2a[i][0][0]) + "; with equally spaced nodes")
		plt.show()

	'''
	Part b
	'''

	q2b = np.zeros(shape=(3, 5, points+1))	# Question 2a, dimensions: n-value; input type; data point
	for i in range(3):
		n = 5 + (i*5)
		for j in range(points+1):
			q2b[i][0][j] = n
			x = a_2+(j*(interval_size_2/points))
			q2b[i][1][j] = x
			q2b[i][2][j] = f2(x)
			q2b[i][3][j] = pn(x, i+3)
			q2b[i][4][j] = en(x, i+3)

	for i in range(3):
		plt.plot(q2b[i,1,:], q2b[i,2,:])
		plt.plot(q2b[i,1,:], q2b[i,3,:])
		plt.xlabel("x value")
		plt.title("Plot for n = " + str(q2b[i][0][0]) + "; with Chebyshev nodes")
		plt.show()

		plt.plot(q2b[i,1,:], q2b[i,4,:])
		plt.xlabel("x value")
		plt.ylabel("en")
		plt.title("Plot for n = " + str(q2b[i][0][0]) + "; with Chebyshev nodes")
		plt.show()


	'''
	Question 3
	'''

	# First must construct the constants in A for the different splines; note global --> A = np.zeros(shape=(n3,2))

	dx = interval_size_3/n3
	for i in range(n3):
		x1 = i*dx
		x2 = x1+dx
		dy = f3(x2)-f3(x1)
		m = dy/dx
		c = f3(x1)-(m*x1) # y=mx+c --> c=y-mx
		A[i][0] = m
		A[i][1] = c


	q3 = np.zeros(shape=(4, points+1))	# x, f(x), S(n, x), absdif_f_S; intialised to 0
	for i in range(points):	# treat x = 1 seperately
		x = i/points
		q3[0][i] = x
		q3[1][i] = f3(x)
		q3[2][i] = S(n3, x)
		q3[3][i] = absdif_f_S(n3, x)

	q3[0][points] = 1
	q3[1][points] = f3(1)
	q3[2][points] = q3[1][points]
	q3[3][points] = absdif_f_S(n3, 1)

	plt.plot(q3[0,:], q3[1,:])
	plt.plot(q3[0,:], q3[2,:])
	plt.xlabel("x value")
	plt.title("Plot for function and linear polynomial approximation at n = " + str(n3))
	plt.show()

	plt.plot(q3[0,:], q3[3,:])
	plt.xlabel("x value")
	plt.ylabel("absolute difference")
	plt.title("Absolute difference plot at n = " + str(n3))
	plt.show()

	print("The maximum absolute difference of all differences is: " + str(np.amax(q3[3,:])))
	print("This difference occurs at x = " + str(q3[0,np.argmax(q3[3,:])]))


if __name__ == "__main__":
	main()