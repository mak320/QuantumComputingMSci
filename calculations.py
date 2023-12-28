import numpy as np

# Define the matrices
t1 = np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]])
t2 = np.array([[0, 0, 1j], [0, 0, 0], [-1j, 0, 0]])
t3 = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]])

# print(t1 @ t1)
# print()
# print(t2 @ t2)
# print()
# print(t3 @ t3)
# print()
# print("------------")
# print(t1 @ t2)
# print()
# print(t1 @ t3)
# print()
# print(t2 @ t3)
# print()
# print("------------")
# print(t2 @ t1)
# print()
# print(t3 @ t1)
# print()
# print(t3 @ t2)
# print()


# print((-t1+ t3))


A = np.ones(10)
B = np.arange(0, 6, 1)

print(np.kron(A, B))


L= np.linspace(-1,1, 10)
print(L)