fast_lr : fast-lr.o csv.o generate.o fit.o predict.o ridge.o
	gcc -Wall -o fast-lr fast-lr.o csv.o generate.o fit.o predict.o ridge.o -lgsl -lcblas -lgslcblas

fast-lr.o : fast-lr.c csv.h generate.h fit.h predict.h ridge.h
	gcc -Wall -c csv.c generate.c fit.c predict.c fast-lr.c ridge.c -lgsl -lcblas -lgslcblas

csv.o : csv.c csv.h
	gcc -Wall -c csv.c -lm -lgsl -lcblas -lgslcblas

generate.o : generate.c generate.h
	gcc -Wall -c generate.c

fit.o : fit.c fit.h
	gcc -Wall -c fit.c

ridge.o : ridge.c ridge.h
	gcc -Wall -c ridge.c

predict.o : predict.c predict.h
	gcc -Wall -c predict.c

clean :
	rm fast-lr.o csv.o generate.o fit.o predict.o ridge.o
