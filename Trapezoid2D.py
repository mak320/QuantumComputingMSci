import numpy as np
from scipy.integrate import dblquad

def function_to_integrate(x, v):
    # Define the function to be integrated
    return x**2 + v**2

def double_integral_trapezoid(x_max, v_max, a, num_points_x, num_points_v):
    # Parameters
    x_values = np.linspace(-x_max, a, num_points_x)
    v_values = np.linspace(-v_max, v_max, num_points_v)

    # Trapezoid rule integration
    result = 0.0

    delta_x = (a + x_max) / num_points_x
    delta_v = (2 * v_max) / num_points_v  # Corrected bounds for v

    # Corner terms
    result += 0.25 * delta_x * delta_v * (
        function_to_integrate(-x_max, -v_max) + function_to_integrate(a, -v_max) +
        function_to_integrate(-x_max, v_max) + function_to_integrate(a, v_max)
    ) 

    # Edge terms
    result += 0.5 * delta_x * delta_x * (
        np.sum([function_to_integrate(x_i, -v_max) for x_i in x_values[1:-1]]) +
        np.sum([function_to_integrate(x_i, v_max) for x_i in x_values[1:-1]])
    )  

    result += 0.5 * delta_x * delta_v * (
        np.sum([function_to_integrate(-x_max, v_j) for v_j in v_values[1:-1]]) +
        np.sum([function_to_integrate(a, v_j) for v_j in v_values[1:-1]])
    ) 

    # Interior terms
    result += delta_x * delta_v * np.sum([
        function_to_integrate(x_i, v_j) for x_i in x_values[1:-1] for v_j in v_values[1:-1]
    ])

    return result

# Example usage
x_max = 5.0
v_max = 3.0
a_values = [1.0, 2.0, 3.0]  # Different values for parameter a

for a in a_values:
    result_trapezoid = double_integral_trapezoid(x_max, v_max, a, num_points_x=100, num_points_v=100)
    
    # Using scipy's dblquad for comparison
    result_scipy, _ = dblquad(function_to_integrate, -x_max, a, lambda v: -v_max, lambda v: v_max)

    print(f"For a = {a}, Trapezoidal Result: {result_trapezoid}, Scipy Result: {result_scipy}")
    print(result_trapezoid/result_scipy)
