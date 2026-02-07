g3 0 1 0	# problem ex1222
 3 3 1 0 0	# vars, constraints, objectives, ranges, eqns
 1 1	# nonlinear constraints, objectives
 0 0	# network constraints: nonlinear, linear
 1 1 1	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 1 0 0 0 0	# discrete variables: binary, integer, nonlinear (b,c,o)
 6 2	# nonzeros in Jacobian, gradients
 3 2	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
b
0 .2 1
0 -2.22554 -1
0 0 1
x2
0 .2
1 -1
r
1 0
1 -1
1 0
C0
o16
o44
o0
n-.2
v0
C1
n0
C2
n0
O0 0
o0
o2
n5
o5
o0
n-.5
v0
n2
n.8
k2
2
4
J0 2
0 0
1 -1
J1 2
1 1
2 1.1
J2 2
0 1
2 -1.2
G0 2
0 0
2 -.7
