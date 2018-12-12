# Fast Linear Regression in C

Clone and install
```
git clone https://github.com/alessandrobessi/fast-linear-regression.git
cd fast-linear-regression
make
```

Usage:
1) Generate sample data:
```
./fast-lr generate [num_features] [num_examples]
```

2) Estimate coefficients:
```
./fast-lr estimate [path_to_csv_file]
```

Example:
```
./fast-lr generate 2000 100000
./fast-lr estimate sample_train.csv
```
