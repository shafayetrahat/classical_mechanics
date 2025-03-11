# Classical Mechanics 
## Project for KEM382

### Files
harmonic_oscillator_1D.py - simulates two-particle harmonic oscillation in 1D
harmonic_oscillator_2D.py - simulates two-particle harmonic oscillation in 2D
lennard_jones.py - simulates interactions between particles using the Lennard-Jones potential with and without PBCs
requiremets.txt - a list of packages and libraries needed to run the programs


### Installing extensions
In virtual environment, the necessary requirements can be installed with the command
```
pip3 install -r requirements.txt
```

### Running the program
Certain parameters must be provided through command line.
```
harmonic_oscillator_1D.py
python <file name> --time <simulation time> --dt <time step>
Example: python harmonic_oscillator_1D.py --time 30 --dt 0.01

harmonic_oscillator_2D.py
python <file name> --time <simulation time> --dt <time step>
Example: python harmonic_oscillator_2D.py --time 30 --dt 0.01

lennard_jones.py
python lennard_jones.py -h 
usage: lennard_jones.py [-h] [--steps STEPS] [--density DENSITY] 
[--file_pbc FILE_PBC] [--file_no_pbc FILE_NO_PBC] [--dt DT] [--N N]

OR with default parameters
python lennard_jones.py 
```
## Simulation Videos:
### 1D harmonic oscillator
https://github.com/user-attachments/assets/bc0cfda3-b037-4b17-a27e-17aded27f450

### 2D harmonic oscillator


https://github.com/user-attachments/assets/67ffa22d-381a-4554-a3c2-fa7d5e880a76



### Lennard Jones simulation


https://github.com/user-attachments/assets/c451f896-390e-42df-8533-bac3e9de6347


