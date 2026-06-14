$title 0/1 Knapsack MIP (scalar) -- discopt GAMS-link smoke model
* Maximize selected value subject to a capacity. Known optimum: obj = 420
* (select items 1..4, total weight 100). Scalar form so it round-trips
* through discopt's from_gams() reader as well as the GAMS solver link.

Binary Variables y1, y2, y3, y4, y5 "select item" ;
Free Variable obj "total value" ;

Equations objective, weight_limit ;

objective..    obj =e= 60*y1 + 100*y2 + 120*y3 + 140*y4 + 160*y5 ;
weight_limit.. 10*y1 + 20*y2 + 30*y3 + 40*y4 + 50*y5 =l= 100 ;

Model knapsack / all / ;
Solve knapsack using MIP maximizing obj ;
