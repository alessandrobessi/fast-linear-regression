# Fast Linear Regression in C

Clone and install
```
git clone https://github.com/alessandrobessi/fast-linear-regression.git
cd fast-linear-regression
make
```

Usage:
1) Generate sample data. The following command produces two csv files: 
- X_train.csv: a matrix of dimensions (num_examples, num_features) with training examples
- y_train.csv: a matrix of dimensions (num_examples, 1) with training labels
```
./fast-lr generate [num_features] [num_examples]
```


2) Estimate coefficients. The following command estimates the linear regression coefficients starting from two csv file, one containing the training examples and the other one containing the training labels.
```
./fast-lr estimate [X_train_csv_file] [y_train_csv_file]
```

Example:
```
./fast-lr generate 2000 100000
./fast-lr estimate X_train.csv y_train.csv
```
