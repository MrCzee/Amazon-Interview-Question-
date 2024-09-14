import numpy as np

# Constants
h = 40  # height of the poles in feet
sag = 75  # sag in the middle of the cable in feet
d = 150  # distance between the poles in feet

# Catenary function and its derivative
def catenary_eq(a):
    return a * np.cosh(d / (2 * a)) - sag - h

#derivative of Catnery
def catenary_eq_derivative(a):
    return np.cosh(d / (2 * a)) - (d / (2 * a)) * np.sinh(d / (2 * a))

# Newton's Method to approximate a
def newtons_method(f, f_prime, initial_guess, tol=1e-5, max_iter=20):
    a = initial_guess
    iterations = []
    for i in range(max_iter):
        fa = f(a)
        iterations.append((i, a, fa))
        if abs(fa) < tol: 
            break
        # Correct the update step by subtracting f(a) / f'(a) from a
        a -= fa / f_prime(a)  # update
    return a, iterations, i + 1  # Return the number of iterations (i + 1 since i is 0-indexed)

# Initial guess for a
initial_guess = d / (2 * np.arccosh((sag + h) / h))

# Approximating a using Newton's method
a_approx, iteration_data, num_iterations = newtons_method(catenary_eq, catenary_eq_derivative, initial_guess)

# Length of the cable
def cable_length(a):
    return 2 * a * np.sinh(d / (2 * a))

length_of_cable = cable_length(a_approx)

# Display the results
print("\n\n\n==========================================")

# Display the results
print(f"Approximated a: {a_approx:.5f}")
print(f"Length of the cable: {length_of_cable:.5f} feet")
print(f"Number of iterations: {num_iterations}")
print("\nIteration data:")
for iteration in iteration_data:
    print(f"Iteration {iteration[0]}: a = {iteration[1]:.5f}, f(a) = {iteration[2]:.5f}")
