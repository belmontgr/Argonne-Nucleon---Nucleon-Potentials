# Argonne Nucleon - Nucleon Potentials in Python

This repository contains a Python port of the FORTRAN function written by the Argonne team that calculates the various Nucleon - Nucleon potentials introduced by the mean. More info and the FORTRAN code can be found  in Argonne's official site: https://www.phy.anl.gov/theory/research/av18/

It is a simple port done by me to possibly help the community.

# Description
The Argonne potentials, such as V18, V8', V4', etc., are Nucleon - Nucleon potentials commonly used in nuclear physics calculations.

# Parameters
The av18pw function accepts the following parameters:

1. **lpot**: Switch for potential choice. 
   - **Argonne potentials**:
     - `1`: av18
     - `2`: av8'
     - `3`: av6'
     - `4`: av4'
     - `5`: avx'
     - `6`: av2'
     - `7`: av1'
     - `8`: modified av8'
   - **Super-Soft Core (C) potentials**:
     - `101`: sscc v14
     - `102`: sscc v8'
     - `108`: modified sscc v8'
     
2. **l**: Orbital angular momentum of the pair. Can be values like 0, 1, 2, ....

3. **s**: Total spin of the pair. Can be 0 or 1.

4. **j**: Total angular momentum of the pair. Can be values like 0, 1, 2, ....

5. **t**: Total isospin of the pair. Can be 0 or 1.

6. **t1z**: Isospin of particle 1. 
   - Use `1` for proton 
   - Use `-1` for neutron

7. **t2z**: Isospin of particle 2. 
   - Use `1` for proton 
   - Use `-1` for neutron

8. **r**: Separation in femtometers (fm).

9. **Output**: The function returns the potential in MeV as a 2x2 array. This includes all strong and electromagnetic terms.




# Usage
```python
# Define parameters
lpot = 1
l = 0
s = 0
j = 0
t = 1
t1z = -1
t2z = -1
r = 1.5  # for example

# Get the potential
vpw = av18pw(lpot, l, s, j, t, t1z, t2z, r)
print(vpw)
```
# Python Script
The example code included in the Python Script calculates and plots the Argonne V18 potential in the ${}^1{S_0}$ channel for Neutron Matter

# References
1. "Accurate nucleon-nucleon potential with charge-independence breaking" R. B. Wiringa, V. G. J. Stoks, and R. Schiavilla, Physical Review C51, 38 (1995) https://journals.aps.org/prc/abstract/10.1103/PhysRevC.51.38 https://arxiv.org/abs/nucl-th/9408016
2. "Quantum Monte Carlo calculations of nuclei with A<=7" B. S. Pudliner, V. R. Pandharipande, J. Carlson, Steven C. Pieper, and R. B. Wiringa, Physical Review C56, 1720 (1997) https://journals.aps.org/prc/abstract/10.1103/PhysRevC.56.1720 https://arxiv.org/abs/nucl-th/9705009
3. "Evolution of Nuclear Spectra with Nuclear Forces" R. B. Wiringa and Steven C. Pieper, Physical Review Letters 89, 182501 (2002) https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.89.182501 https://arxiv.org/abs/nucl-th/0207050

# License
This project is licensed under the MIT License. See the LICENSE file for more details.

