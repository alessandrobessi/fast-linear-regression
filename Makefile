fast-lr: fast-lr.c csv.c csv.h generate.c generate.h fit.c fit.h
	gcc -Wall fit.c generate.c csv.c fast-lr.c -o fast-lr -lgsl -lcblas -lgslcblas

clean:
	rm fast-lr X_train.csv y_train.csv
