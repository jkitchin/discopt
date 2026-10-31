var x >= 0, <= 10;  var y >= 0;
minimize o: (x-1)^2 + (y-1)^2;
subject to cap: y <= 10;
subject to c: 0 <= y <= Infinity complements x;
