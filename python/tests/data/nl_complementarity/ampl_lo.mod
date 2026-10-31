var x >= 3;  var y >= -10, <= 10;
minimize o: (x-3)^2 + (y-1)^2;
subject to cap: x <= 10;
subject to c: 3 <= x <= Infinity complements y;
