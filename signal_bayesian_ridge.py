from sklearn.linear_model import BayesianRidge


clf = BayesianRidge()
clf.fit(train_data[all_vars], train_data['MEDV'])
BR_all_predictions = clf.predict(test_data[all_vars])
test_data['BR_all'] = BR_all_predictions
BR_all_RMSE = mean_squared_error(train_data.MEDV[:150], test_data.BR_all[:150])**0.5

RMSE_values["BR_all"] = BR_all_RMSE
