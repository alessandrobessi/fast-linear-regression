fast-lr: fast-lr.c csv.c csv.h generate.c generate.h
	gcc -Wall generate.c csv.c fast-lr.c -o fast-lr -lgsl -lcblas -lgslcblas

clean:
	rm fast-lr sample_train.csv
