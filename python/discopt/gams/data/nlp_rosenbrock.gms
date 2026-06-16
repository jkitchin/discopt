$title Rosenbrock NLP -- discopt GAMS-link smoke model
* Nonconvex banana valley. Known global optimum: obj = 0 at (x1, x2) = (1, 1).
* Variables bounded to keep spatial branch-and-bound bounded.

Free Variables x1, x2, obj ;

x1.lo = -2.0 ;  x1.up = 2.0 ;
x2.lo = -2.0 ;  x2.up = 2.0 ;
x1.l  = -1.0 ;  x2.l  = 1.0 ;

Equations rosenbrock ;
rosenbrock.. obj =e= sqr(1 - x1) + 100 * sqr(x2 - sqr(x1)) ;

Model rosen / all / ;
Solve rosen using NLP minimizing obj ;
