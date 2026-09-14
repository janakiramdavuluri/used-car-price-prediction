from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

categorical_features = ["name","fuel","seller_type","transmission","owner"]

numerical_features = ["km_driven","vehicle_age"]

preprocessor = ColumnTransformer([("cat",OneHotEncoder(handle_unknown="ignore",min_frequency=2),categorical_features),
                                  ("num",StandardScaler(),numerical_features)])
