from sympy import *
import sympy as sp
from sympy.abc import *
import numpy as np

def lprint(expr, pre='', add=True,):
    latex_expr = sp.latex(expr)
    if add:
        with open('content.tex', 'a+') as f:
            f.write(f'$$\n{pre} {latex_expr}\n$$')
    else:
        with open('content.tex', 'w') as f:
            f.write(f'$$\n{pre} {latex_expr}\n$$\n')

def tprint(text, add=True):
    if add:
        with open('content.tex', 'a+') as f:
            f.write(f'\n\n$$\\quad\\textbf{{{text}}}\n$$\n\n')
    else:
        with open('content.tex', 'w') as f:
            f.write(f'\n\n$$\\quad\\textbf{{{text}}}\n$$\n\n')

lprint(' ', add=False)

#  i = Idx('i', (1, 3))
#  j = Idx('j', (1, 3))
#  k = Idx('k', (1, N))

#  l = Idx('l', (1, 3))
#  m = Idx('m', (1, 3))
#  n = Idx('n', (1, 3))

O = IndexedBase('O')
V = IndexedBase('V')
S = IndexedBase('S')

s = Sum(
        (S[i,j] * (V[j] - O[j]))**2,
        (j, 1, 3)
        )

# Declaro e y la simetrizo
e = Sum(
        s - g**2,
        (i, 1, 3)
        )
e = e.subs(S[2,1], S[1,2]).subs(S[3,1], S[1,3]).subs(S[3,2], S[2,3])
E = e**2


Theta = [O[i] for i in range(1, 4)] + [S[i,j] for i in range(1,4)
                                              for j in range(1,4)
                                              if i <= j]
J = zeros(9, 1)
H = zeros(9, 9)
for i, theta_i in enumerate(Theta):
    E_diff_theta_i = E.diff(theta_i)
    J[i] = (E_diff_theta_i)
    for j, theta_j in enumerate(Theta):
        E_diff_theta_i_theta_j = E_diff_theta_i.diff(theta_j)
        H[i, j] = E_diff_theta_i_theta_j

#  E_diff_O_n = E.diff(Theta)
#  E_diff_S_lm = E.diff(S[l,m])

# Declaramos J
#  J = [O[i] for i in range(1, 4)] + [S[i,j] for i in range(1,4)
                                          #  for j in range(1,4)
                                          #  if i <= j]
#  lprint(E_diff_O_n, '\pdv{E}{O_n}=')
#  lprint(E_diff_S_lm, '\pdv{E}{S_{lm}}=')
lprint(J, 'J=')
#  lprint(H, 'H=')
