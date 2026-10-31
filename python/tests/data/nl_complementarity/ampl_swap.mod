var x >= 0;  var y >= -10, <= 10;
minimize o: x^2 + (y+1)^2;
subject to cap: x <= 10;
subject to c: 0 <= x <= Infinity complements y;
