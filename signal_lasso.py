from sklearn.linear_model import Lasso


clf = Lasso(alpha = 0.1)
clf.fit(train_data[all_vars], train_data['MEDV'])
lasso_all_predictions = clf.predict(test_data[all_vars])
test_data['lasso_all'] = lasso_all_predictions
lasso_all_RMSE = mean_squared_error(train_data.MEDV[:150], test_data.lasso_all[:150])**0.5


RMSE_values["lasso_all"] = lasso_all_RMSE


#
