$title Transport LP (2x2, original) -- discopt GAMS-link smoke model
* A small, self-contained transportation LP authored for this test corpus
* (not derived from any GAMS-distributed example). Known optimum: z = 75,
* shipping x(p1,m1)=20, x(p2,m1)=5, x(p2,m2)=15.

Sets
    i "plants"  / p1, p2 /
    j "markets" / m1, m2 / ;

Parameter cap(i) "plant capacity" / p1 20, p2 30 / ;
Parameter dem(j) "market demand"  / m1 25, m2 15 / ;

Table c(i,j) "unit shipping cost"
          m1   m2
    p1     2    3
    p2     4    1 ;

Positive Variable x(i,j) "shipments" ;
Free Variable z "total shipping cost" ;

Equations cost, supply(i), demand(j) ;

cost..      z =e= sum((i,j), c(i,j) * x(i,j)) ;
supply(i).. sum(j, x(i,j)) =l= cap(i) ;
demand(j).. sum(i, x(i,j)) =g= dem(j) ;

Model transport / all / ;
Solve transport using LP minimizing z ;
