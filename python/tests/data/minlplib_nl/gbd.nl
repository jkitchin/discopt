g3 0 1 0	# problem gbd
 4 4 1 0 0	# vars, constraints, objectives, ranges, eqns
 0 1	# nonlinear constraints, objectives
 0 0	# network constraints: nonlinear, linear
 0 1 0	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 3 0 0 0 0	# discrete variables: binary, integer, nonlinear (b,c,o)
 12 4	# nonzeros in Jacobian, gradients
 3 2	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
b
0 .2 1
0 0 1
0 0 1
0 0 1
x1
0 .2
r
1 0
1 0
2 2
2 2
C0
n0
C1
n0
C2
n0
C3
n0
O0 0
o2
n5
o5
v0
n2
k3
2
5
9
J0 3
0 3
1 -1
2 -1
J1 3
0 -1
2 .1
3 .25
J2 3
1 1
2 1
3 1
J3 3
1 1
2 1
3 2
G0 4
0 0
1 1
2 1
3 1
