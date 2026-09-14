import pandas as pd
from preprocess import preprocessor
from train import train_models
def predict_price(input_data,x_train_processed,y_train):
    """
    Predict the selling price of a used car
    using the Random Forest model.
    """

    input_processed=preprocessor.transform(input_data)

    models=train_models(x_train_processed,y_train)

    prediction=models["Random Forest"].predict(input_processed)

    return prediction[0]

#commit