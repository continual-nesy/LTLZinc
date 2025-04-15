p_0(X, Y, Z) :- value(X, V1), value(Y, V2), value(Z, V3), V1 =\= V2, V2 =\= V3, V3 =\= V1.
p_1(X, Y, Z) :- value(X, V1), value(Y, V2), value(Z, V3), V1 < V2 + V3.

query(p_0(var_x, var_y, var_z)).
query(p_1(var_x, var_y, var_z)).