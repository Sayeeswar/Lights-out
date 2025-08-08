# 🔆 Lights Out Solver – Square Grid & Graph Configurations

## 📌 Overview
This project provides a **complete solver** for the *Lights Out* game in both:
- **Square grid** representation (`n × n` board)
- **Custom graph-based** representation

It can:
- Determine whether a configuration is solvable.
- Compute **all possible solvable/unsolvable configurations**.
- Solve the puzzle using **Gaussian elimination over GF(2)**.
- Calculate and display the **null space** for analysis.

---

## ✨ Features
- **Toggle Matrix Generation** – Builds adjacency/toggle matrices for square grids and custom graphs.
- **Augmented Gaussian Elimination (mod 2)** – Performs forward and backward elimination for solving.
- **Null Space Computation (mod 2)** – Finds all solution vectors.
- **Configuration Generator** – Produces random solvable states.
- **Graph Solver** – Works with arbitrary node/neighbor definitions.
- **Square Grid Solver** – Supports any `n × n` Lights Out grid.
- **Full Search Mode** – Counts solvable vs. unsolvable configurations.
- **Step-by-Step Display** – Prints intermediate matrices and solution steps.

---

## 🛠 How It Works
1. **Create Toggle Matrix** – Represents how pressing a light affects others.
2. **Set Initial Configuration** – Random or predefined.
3. **Run Gaussian Elimination (mod 2)** – Solves the system of equations.
4. **Check Null Space** – Determines solvability and alternative solutions.
5. **Output**:
   - Augmented matrix in RREF.
   - Solution vector (lights to toggle).
   - Null space basis.
   - Count of solvable and unsolvable configurations.

---

## 📂 File Structure
Contains all functions:
- `create_toggle_matrix(size)`
- `augmented_gaussian_elimination(matrix, b)`
- `null_space_mod2(A)`
- `generate_solvable_configurations(...)`
- `generate_lights_out_graph()`
- `solve_lights_outgraph()`
- `solve_lights_out(size, initial_configuration)`
- `testing(...)` / `testingg(...)`
- `main()` – Runs example graph & grid solutions.

---

## ▶️ Example Usage
```bash
python Lightsout.py

