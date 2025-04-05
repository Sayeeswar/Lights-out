import numpy as np

def create_toggle_matrix(size):
    matrix = np.zeros((size * size, size * size), dtype=int)
    for i in range(size * size):
        row, col = divmod(i, size)
        matrix[i][i] = 1
        if col > 0: matrix[i][i - 1] = 1  # Left
        if col < size - 1: matrix[i][i + 1] = 1  # Right
        if row > 0: matrix[i][i - size] = 1  # Up
        if row < size - 1: matrix[i][i + size] = 1  # Down
    return matrix

def augmented_gaussian_elimination(matrix, b):
    rows, cols = matrix.shape
    augmented_matrix = np.hstack((matrix, b.reshape(-1, 1))) % 2  # Ensure all entries are in mod 2


    # Forward Elimination (Row Echelon Form)
    for r in range(rows):
        # Pivoting: Swap with a row that has a leading 1 if necessary
        if augmented_matrix[r, r] == 0:
            for i in range(r + 1, rows):
                if augmented_matrix[i, r] == 1:
                    augmented_matrix[[r, i]] = augmented_matrix[[i, r]]  # Swap rows
                    print(f"Swapped row {r + 1} with row {i + 1}")
                    print(augmented_matrix)
                    break  

        # XOR to eliminate 1s below the pivot
        for i in range(r + 1, rows):
            if augmented_matrix[i, r] == 1:  # Only process rows that have a 1 in the pivot column
                augmented_matrix[i] ^= augmented_matrix[r]  # Row XOR operation
                print(f"XOR row {i + 1} with row {r + 1}")
                print(augmented_matrix)

    # Backward Elimination (Reduce to RREF)
    for r in range(rows - 1, -1, -1):
        if augmented_matrix[r, r] == 1:  # Ensure pivot is 1
            for i in range(r - 1, -1, -1):  # Clear values above the pivot
                if augmented_matrix[i, r] == 1:
                    augmented_matrix[i] ^= augmented_matrix[r]
                    print(f"XOR row {i + 1} with row {r + 1} (Back substitution)")
                    print(augmented_matrix)

    return augmented_matrix[:, :-1] % 2, augmented_matrix[:, -1] % 2  # Return matrix and solution in mod 2

import numpy as np

def null_space_mod2(A):
    A = np.array(A) % 2  # Ensure mod 2
    rows, cols = A.shape

    # Transpose A
    A_T = A.T
    print("Transposed Matrix:\n", A_T)

    pivots = []
    free_vars = []
    
    # Identify pivots and free variables
    row_used = set()
    for c in range(rows):  # Use rows now (since we transposed)
        pivot_row = -1
        for r in range(cols):  # Check all columns (original rows)
            if A_T[r, c] == 1 and r not in row_used:
                pivot_row = r
                break

        if pivot_row != -1:
            pivots.append(c)
            row_used.add(pivot_row)
        else:
            free_vars.append(c)

    print("Pivots:", pivots)
    print("Free Variables:", free_vars)

    # Construct null space vectors
    null_space_vectors = []
    for free_var in free_vars:
        null_vector = np.zeros(rows, dtype=int)
        null_vector[free_var] = 1  # Set free variable to 1
        
        # Use transposed matrix to get dependencies
        null_vector[:] = A_T[free_var]  # Directly take the column from A_T

        null_space_vectors.append(null_vector)

    return np.array(null_space_vectors).T if null_space_vectors else np.zeros((rows, 0), dtype=int)




def print_dot_product(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    product_terms = ' + '.join(f"{a}*{b}" for a, b in zip(vec1, vec2))
    print(f"{vec1} ⋅ {vec2} = {product_terms} = {dot_product%2}")#fstrings to print everything that you write in the sprint statemnt while executing functons

def generate_solvable_configurations(matrix, num_configs=10):
    """Generates solvable configurations using column space."""


    # Use a random combination of toggle matrix columns to generate solvable states
    solvable_configs = []
    num_cols = matrix.shape[1]

    for _ in range(num_configs):
        random_coeffs = np.random.randint(0, 2, num_cols)  # Random binary coefficients
        solvable_config = (matrix @ random_coeffs) % 2  # random configuration it the steps that we should take to get a configuration
        print("the solvable config",solvable_config)
        solvable_configs.append(solvable_config)

    return np.array(solvable_configs)

def generate_lights_out_graph():
    graph = {
        1: [2,6],
        2: [3,1],
        3: [2,4],
        4: [3,5],
        5: [6,4],
        6:[1,5]
       }

    n = max(graph.keys())
    matrix = np.zeros((n, n), dtype=int)

    for node, neighbors in graph.items():
        for neighbor in neighbors:
             matrix[node-1][neighbor-1] = 1

   # b = generate_unique_random_configs(5, 2**5)
    b=[0,1,1,1,0,1]
    print(matrix)
    print(b)
    b= np.array(b)

    return matrix, b

def solve_lights_outgraph():
     matrix,b = generate_lights_out_graph()
     echelon_form,solution = augmented_gaussian_elimination(matrix,b)
     echelon_form=np.array(echelon_form)
     print("the echoleon form is ",type(echelon_form))
     nullspace=null_space_mod2(echelon_form)


     print("the intial configurationi is ",b)
     print("The null space is ",nullspace)
     print("inital matrix",matrix)
     return echelon_form,nullspace,sol,b

def testing(b):
    size = 6
    matrix = create_toggle_matrix(size)
    count = 0
    count1 = 0

    for i in range(2**(size * size)):
        initial_configuration = np.array([int(x) for x in format(i, f'0{size*size}b')], dtype=int)
        initial_configuration = initial_configuration.astype(int)  # Correct spelling

        matrix1, solution = augmented_gaussian_elimination(matrix, initial_configuration)
        nullspace = null_space_mod2(matrix1)

        if np.any(nullspace) and np.all(np.dot(nullspace.T, solution) % 2 == 0):
            solvable = np.array(solution)
            count += 1

            for sol in solution:
                sol = np.atleast_1d(sol)  # Ensures sol is at least 1D
                if np.array_equal(b, sol):  # Exact match check
                    print("yes")
                else:
                  print("no")

        else:
            count1 += 1
            
    
    print(count1)


def testingg(b):  
    size = 6  # Grid size
    matrix, b = generate_lights_out_graph()  # Ensure this function exists

    count = 0
    count1 = 0
    count2 = 0  # Fixed declaration

    unsolvable = []  
    solvable = []
    initial_configuration =np.zeros(size,dtype=int)
    initial_configuration=np.array(initial_configuration)
    matrix1, solution = augmented_gaussian_elimination(matrix, initial_configuration)
    nullspace = null_space_mod2(matrix1)
    print(nullspace)
    for i in range(2**size):
        
       
        initial_configuration = np.array([int(x) for x in format(i, f'0{size}b')], dtype=int)
        if np.any(nullspace) and np.all(np.dot(nullspace.T, initial_configuration) % 2 == 0):
            count += 1
            solvable.append(np.array(initial_configuration))  

        elif np.all(nullspace == 0):  # ✅ Correct indentation
            count2 += 1
            solvable.append(np.array(initial_configuration))  

        else:
            count1 += 1
            unsolvable.append(np.array(initial_configuration))  

    print("\nSolvable configurations stored:", len(solvable))
    print("\nTotal unsolvable configurations:", count1)
    print("Total solvable configurations:", count)
    print("Count2:", count2)  # ✅ Ensure this prints before any return

    b = np.array(b).flatten()
    print("\nTHE CONFIGURATIONS THAT ARE UNSOLVABLE\n") 
    for sol in solvable:
        sol = np.array(sol).flatten()  

        print(f"\nComparing:\nInitial (b): {b} (shape: {b.shape})\nSolution (sol): {sol} (shape: {sol.shape})")

        
         
        if np.array_equal(b, sol):  
            print("\n✅ MATCH FOUND!")
           # print("Initial configuration:", b)
            #print("Solvable configuration:", sol)
            

   
    print("\nTHE CONFIGURATIONS THAT ARE SOLVABLE\n") 
    for sol in unsolvable:
        sol = np.array(sol).flatten()  

        print(f"\nComparing:\nInitial (b): {b} (shape: {b.shape})\nSolution (sol): {sol} (shape: {sol.shape})")

        if np.array_equal(b, sol):  
            print("\n✅ MATCH FOUND!")
           # print("Initial configuration:", b)
           # print("Solvable configuration:", sol)
            

    print("\nTotal unsolvable configurations:", count1)
    print("Total solvable configurations:", count)
    print("Count2:", count2)  # ✅ Ensures print before function exit


sol=np.array([1,0,1,1,0,1])
augument,nullspace,sol,b= solve_lights_outgraph()

print(augument)
print(sol)


def solve_lights_out(size, initial_configuration):
    toggle_matrix = create_toggle_matrix(size)

    augmented_matrix, solution_vector = augmented_gaussian_elimination(toggle_matrix, initial_configuration)
    solution_vector = solution_vector % 2  # Ensure binary values
    solution_matrix = solution_vector.reshape((size, size))

    null_space = null_space_mod2(augmented_matrix)
    #is_solvable_flag = np.all(np.dot(augmented_matrix, null_space) % 2 == 0)

    print("Augmented Matrix:")
    print(augmented_matrix)

    print("Solution Steps (Toggle these positions to solve, represented as a step-by-step matrix):")
    for row in solution_matrix:
        print(" ".join(map(str, row)))

    return  augmented_matrix, solution_matrix,null_space

def main():
    # Solve the Lights Out game for a predefined graph
    print("Solving Lights Out for Graph Representation:")
    augment,null, sol,get=solve_lights_outgraph()
    print("Augmented Matrix for Graph:")
    print(augment)
    print("Solution Vector:")
    print(sol)
    testingg(sol)

    # Solve the Lights Out game for an n x n grid
'''size = 4
initial_configuration = np.random.randint(0, 2, size * size)  # Generate a random initial state
b=initial_configuration
    
print("\nSolving Lights Out for Grid Representation:")
x=initial_configuration

augmented_matrix, solution_steps,null_space = solve_lights_out(size, initial_configuration)

    # Compute and print the null space
    nullspace = null_space_mod2(augmented_matrix)
    print_dot_product(x,nullspace)
    print("\nThe null space vectors are:\n", nullspace)
    #testing(b)
   
   

    # Display initial configuration
    print("\nInitial Configuration:")
    for row in initial_configuration.reshape((size, size)):  # Reshape into square format
        print(" ".join(map(str, row)))

    # Print solution steps
    print("\nSteps to Turn Off All Lights:")
    for row in solution_steps:
        print(" ".join(map(str, row)))
'''
if __name__ == "__main__":
     main()
