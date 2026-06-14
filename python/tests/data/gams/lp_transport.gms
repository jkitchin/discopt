$title Transport LP (gamslib trnsport) -- discopt GAMS-link smoke model
* Classic Dantzig transportation problem. Known optimum: z = 153.675.

Sets
    i "canning plants" / Seattle, San_Diego /
    j "markets"        / New_York, Chicago, Topeka / ;

Scalar f "freight in dollars per case per thousand miles" / 90 / ;

Table d(i,j) "distance in thousands of miles"
              New_York  Chicago  Topeka
    Seattle      2.5      1.7     1.8
    San_Diego    2.5      1.8     1.4 ;

Parameter a(i) "capacity of plant i in cases"  / Seattle 350, San_Diego 600 / ;
Parameter b(j) "demand at market j in cases"   / New_York 325, Chicago 300, Topeka 275 / ;

Positive Variables x(i,j) "shipment quantities in cases" ;
Free Variable z "total transportation costs in thousands of dollars" ;

Equations
    cost      "define objective function"
    supply(i) "observe supply limit at plant i"
    demand(j) "satisfy demand at market j" ;

cost..      z =e= sum((i,j), f * d(i,j) * x(i,j) / 1000) ;
supply(i).. sum(j, x(i,j)) =l= a(i) ;
demand(j).. sum(i, x(i,j)) =g= b(j) ;

Model transport / all / ;
Solve transport using LP minimizing z ;
