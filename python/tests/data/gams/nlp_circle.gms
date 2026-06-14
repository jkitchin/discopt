$title Constrained least-distance NLP -- discopt GAMS-link smoke model
* Minimize squared distance from (1, 2) subject to x + y >= 5.
* Convex QP/NLP. Known optimum: obj = 2.0 at (x, y) = (2, 3).

Free Variables x, y, obj ;

Equations obj_def, line ;

obj_def.. obj =e= sqr(x - 1) + sqr(y - 2) ;
line..    x + y =g= 5 ;

Model circle / all / ;
Solve circle using NLP minimizing obj ;
