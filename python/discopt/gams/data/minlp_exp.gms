$title Convex MINLP with exp + binary -- discopt GAMS-link smoke model
* Minimize z = exp(x) - 3*x + 5*y  s.t. x >= y, x in [0, 3], y in {0, 1}.
* Convex relaxation; the binary stays off. Known optimum:
*   z = 3 - 3*ln(3) = -0.2958368660043290 at (x, y) = (ln 3, 0).

Positive Variable x ;
Binary Variable   y ;
Free Variable     z ;

x.up = 3 ;

Equations obj_def, link ;

obj_def.. z =e= exp(x) - 3*x + 5*y ;
link..    x - y =g= 0 ;

Model minlp_exp / all / ;
Solve minlp_exp using MINLP minimizing z ;
