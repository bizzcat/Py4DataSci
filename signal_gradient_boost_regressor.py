from sklearn.ensemble import GradientBoostingRegressor

GBR_strong = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
    max_depth=1, random_state=0, loss='ls')
GBR_strong.fit(train_data[strong_vars], train_data['MEDV'])
GBR_strong_predictions = GBR_strong.predict(test_data[strong_vars])
test_data['GBR_strong'] = GBR_strong_predictions
GBR_strong_RMSE = mean_squared_error(train_data.MEDV[:150], test_data.GBR_strong[:150])**0.5



GBR_all_vars = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
    max_depth=1, random_state=0, loss='ls')
GBR_all_vars.fit(train_data[all_vars], train_data['MEDV'])
GBR_all_predictions = GBR_all_vars.predict(test_data[all_vars])
test_data['GBR_all'] = GBR_all_predictions
GBR_all_RMSE = mean_squared_error(train_data.MEDV[:150], test_data.GBR_all[:150])**0.5



GBR_weak_vars = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
    max_depth=1, random_state=0, loss='ls')
GBR_weak_vars.fit(train_data[weak_vars], train_data['MEDV'])
GBR_weak_predictions = GBR_weak_vars.predict(test_data[weak_vars])
test_data['GBR_weak'] = GBR_weak_predictions
GBR_weak_RMSE = mean_squared_error(train_data.MEDV[:150], test_data.GBR_weak[:150])**0.5



GBR_positive_vars = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
    max_depth=1, random_state=0, loss='ls')
GBR_positive_vars.fit(train_data[positive_vars], train_data['MEDV'])
GBR_positive_predictions = GBR_positive_vars.predict(test_data[positive_vars])
test_data['GBR_positive'] = GBR_positive_predictions
GBR_positive_RMSE = mean_squared_error(train_data.MEDV[:150], test_data.GBR_positive[:150])**0.5



GBR_negative_vars = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
    max_depth=1, random_state=0, loss='ls')
GBR_negative_vars.fit(train_data[negative_vars], train_data['MEDV'])
GBR_negative_predictions = GBR_negative_vars.predict(test_data[negative_vars])
test_data['GBR_negative'] = GBR_negative_predictions
GBR_negative_RMSE = mean_squared_error(train_data.MEDV[:150], test_data.GBR_negative[:150])**0.5



RMSE_values["GBR_all"] = GBR_all_RMSE
RMSE_values["GBR_strong"] = GBR_strong_RMSE
RMSE_values["GBR_weak"] = GBR_weak_RMSE
RMSE_values["GBR_positive"] = GBR_positive_RMSE
RMSE_values["GBR_negative"] = GBR_negative_RMSE
