# Fast Linear Regression in C

Clone and install
```
git clone https://github.com/alessandrobessi/fast-linear-regression.git
cd fast-linear-regression
make
```

Usage:
1) Generate sample data. The following command produces a csv file containing a matrix of dimensions (num_examples, num_features + 1), where the first column represents the dependent variable.
```
./fast-lr generate [num_features] [num_examples]
```


2) Estimate coefficients. The following command estimates the linear regression coefficients starting from a csv where the first columns represents the dependent variable.
```
./fast-lr estimate [path_to_csv_file]
```

Example:
```
./fast-lr generate 2000 100000
./fast-lr estimate sample_train.csv
```
