p_0(V, W, X, Y, Z) :- value(Y, V1), value(Z, V2), V1 < V2.
p_1(V, W, X, Y, Z) :- value(V, V1), value(W, V2), value(X, V3), V1 =:= V2, V2 =:= V3.

query(p_0(var_v, var_w, var_x, var_y, var_z)).
query(p_1(var_v, var_w, var_x, var_y, var_z)).