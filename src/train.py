from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
# Create the regression models
linear_reg=LinearRegression()
ridge_reg=Ridge(alpha=10)
random_forest=RandomForestRegressor(n_estimators=100,random_state=42)
extra_tree = ExtraTreesRegressor(n_estimators=100,random_state=42)
def train_models(x_train_processed,y_train):
    """Train all regression models."""
    linear_reg.fit(x_train_processed,y_train)
    ridge_reg.fit(x_train_processed,y_train)
    random_forest.fit(x_train_processed,y_train)
    extra_tree.fit(x_train_processed,y_train)
    return {
            "Linear Regression":linear_reg,
            "Ridge Regression":ridge_reg,
            "Random Forest":random_forest,
            "Extra Trees":extra_tree
        }