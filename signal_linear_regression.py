from sklearn.linear_model import LinearRegression


LR_strong = LinearRegression()
LR_strong.fit(train_data[strong_vars], train_data['MEDV'])
LR_strong_predictions = LR_strong.predict(test_data[strong_vars])
test_data['LR_strong'] = LR_strong_predictions
LR_strong_RMSE = mean_squared_error(train_data.MEDV[:150], test_data.LR_strong[:150])**0.5



LR_all_vars = LinearRegression()
LR_all_vars.fit(train_data[all_vars], train_data['MEDV'])
LR_all_predictions = LR_all_vars.predict(test_data[all_vars])
test_data['LR_all'] = LR_all_predictions
LR_all_RMSE = mean_squared_error(train_data.MEDV[:150], test_data.LR_all[:150])**0.5



LR_weak_vars = LinearRegression()
LR_weak_vars.fit(train_data[weak_vars], train_data['MEDV'])
LR_weak_predictions = LR_weak_vars.predict(test_data[weak_vars])
test_data['LR_weak'] = LR_weak_predictions
LR_weak_RMSE = mean_squared_error(train_data.MEDV[:150], test_data.LR_weak[:150])**0.5



LR_positive_vars = LinearRegression()
LR_positive_vars.fit(train_data[positive_vars], train_data['MEDV'])
LR_positive_predictions = LR_positive_vars.predict(test_data[positive_vars])
test_data['LR_positive'] = LR_positive_predictions
LR_positive_RMSE = mean_squared_error(train_data.MEDV[:150], test_data.LR_positive[:150])**0.5



LR_negative_vars = LinearRegression()
LR_negative_vars.fit(train_data[negative_vars], train_data['MEDV'])
LR_negative_predictions = LR_negative_vars.predict(test_data[negative_vars])
test_data['LR_negative'] = LR_negative_predictions
LR_negative_RMSE = mean_squared_error(train_data.MEDV[:150], test_data.LR_negative[:150])**0.5


RMSE_values["LR_all"] = LR_all_RMSE
RMSE_values["LR_strong"] = LR_strong_RMSE
RMSE_values["LR_weak"] = LR_weak_RMSE
RMSE_values["LR_positive"] = LR_positive_RMSE
RMSE_values["LR_negative"] = LR_negative_RMSE
