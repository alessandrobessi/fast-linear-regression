fast-lr: fast-lr.c csv.c csv.h
	gcc -Wall csv.c fast-lr.c -o fast-lr -lgsl -lcblas -lgslcblas