p_0(W, X, Y, Z) :- value(W, V1), value(X, V2), value(Y, V3), value(Z, V4), V1 + V2 =:= V3 + V4.

query(p_0(var_w, var_x, var_y, var_z)).