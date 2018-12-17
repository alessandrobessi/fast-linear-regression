# Fast Linear Regression in C

## Clone and install
```
git clone https://github.com/alessandrobessi/fast-linear-regression.git
cd fast-linear-regression
make
make clean
```

## Usage:
1) **Generate sample data.** The following command produces two csv files: 
- X_train.csv: a matrix of dimensions (num_examples, num_features) with training examples
- y_train.csv: a matrix of dimensions (num_examples, 1) with training labels
```
./fast-lr generate [num_features] [num_examples]
```


2) **Fit a linear regression model.** The following command estimates the linear regression coefficients starting from two csv file, one containing the training examples and the other one containing the training labels.
```
./fast-lr fit [X_train_csv_file] [y_train_csv_file] [--verbose]
```

3) **Predict.** The following command predicts y values using two csv file, one containing training examples and the other one containing linear regression coefficients.
```
./fast-lr predict [X_train_csv_file] [beta_csv_file] [--verbose]
```

#### Example:
```
./fast-lr generate 2000 100000
./fast-lr fit X_train.csv y_train.csv --verbose
./fast-lr predict X_train.csv beta.csv --verbose
```
