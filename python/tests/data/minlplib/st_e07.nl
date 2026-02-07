g3 0 1 0	# problem st_e07
 10 7 1 0 5	# vars, constraints, objectives, ranges, eqns
 3 0	# nonlinear constraints, objectives
 0 0	# network constraints: nonlinear, linear
 3 0 0	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0	# discrete variables: binary, integer, nonlinear (b,c,o)
 26 5	# nonzeros in Jacobian, gradients
 3 3	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
b
0 0 100
0 0 200
0 1 3
0 0 300
0 0 300
0 0 100
0 0 300
0 0 100
0 0 200
0 0 200
x1
2 1
r
1 0
1 0
4 0
4 0
4 0
4 0
4 0
C0
o2
v2
v0
C1
o2
v2
v1
C2
o16
o2
v2
o0
v0
v1
C3
n0
C4
n0
C5
n0
C6
n0
O0 0
n0
k9
4
8
11
13
15
17
18
21
24
J0 4
0 0
2 0
5 -2.5
7 2
J1 4
1 0
2 0
8 2
9 -1.5
J2 5
0 0
1 0
2 0
3 3
4 1
J3 4
0 -1
1 -1
3 1
4 1
J4 3
0 1
5 -1
7 1
J5 3
1 1
8 1
9 -1
J6 3
6 -1
7 1
8 1
G0 5
3 6
4 16
5 -9
6 10
9 -15
