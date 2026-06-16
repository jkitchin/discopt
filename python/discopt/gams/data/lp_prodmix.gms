$title Product-mix LP (scalar) -- discopt GAMS-link smoke model
* Textbook product mix. Known optimum: profit = 36 at (x, y) = (2, 6).

Positive Variables x, y ;
Free Variable profit ;

Equations obj_def, c_x, c_y, c_xy ;

obj_def.. profit =e= 3*x + 5*y ;
c_x..     x         =l= 4 ;
c_y..     2*y       =l= 12 ;
c_xy..    3*x + 2*y =l= 18 ;

Model prodmix / all / ;
Solve prodmix using LP maximizing profit ;
