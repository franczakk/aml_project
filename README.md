# AML project 1

## Installation
To access the code, clone the repository:
```bash
git clone https://github.com/kubek22/aml_project
```
Then, navigate to project directory and install dependencies:
```pycon
pip install -r requirements.txt 
```

## Instruction 
The logistic regression model is stored in file logistic_regression.py.
The model can be imported by referring to it:
```python
from logistic_regression import LogRegCCD
lr = LogRegCCD()
```

The most basic way of training is to use `fit` method:
```python
lr.fit(iterations, df_train, y_train, alpha, lmbda)
```
This will train the model with fixed parameters for a given number of iterations.\
The model standardizes the input data internally, so it is not required to do it manually.\
The input pair `df_train`, `y_train` may be of type `np.array`, `pd.Series` of `pd.DataFrame`.

Alternatively, to select the best `lmbda` parameter, you may use:
```python
lr.plot(measure, df_train, y_train, df_test, y_test, alpha)
```

or analyze coefficients' values:
```python
lr.plot_coefficients(df_train, y_train, alpha)
```
In both methods above, the model is trained on a decreasing path of `lmbda` values.

You access the coefficients using:
```python
coefs = lr.beta
```

You can compute the probabilities and predicted classes using:
```python
proba = lr.predict_proba(df_test)
pred = lr.predict(df_test)
```

You can get the maximal `lmbda` value using:
```python
lambda_max = lr.compute_lambda_max(df_train, y_train, alpha)
```