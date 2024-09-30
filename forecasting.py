import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
import mlflow
import mlflow.pyfunc
import logging

# Set up the logging system
logging.basicConfig(level=logging.WARN) # Warning on errors will be shown only.
logger = logging.getLogger(__name__)

def prep_store_data(df: pd.DataFrame, store_id: int = 4, store_open: int = 1) -> pd.DataFrame:
    df['Date'] = pd.to_datetime(df['Date'])
    df.rename(columns= {'Date': 'ds', 'Sales': 'y'}, inplace=True)
    
    # Filter the rows based on 'Store' and 'Open'
    df_store = df[
        (df['Store'] == store_id) &\
        (df['Open'] == store_open)
    ].reset_index(drop=True)
    return df_store.sort_values('ds', ascending=True) 
        
        
class ProphetWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
        
        # Call the constructor of the parent class (PythonModel), ensuring any necessary initialization from the parent class is also done.
        super().__init__()

    def load_context(self, context):
        from prophet import Prophet

        return

    def predict(self, context, model_input):
        # Create a new DataFrame that contains future dates for which you want to make predictions using the Prophet model.
        future = self.model.make_future_dataframe(periods=model_input["periods"][0])
        
        # Use the Prophet model to make predictions based on the created future DataFrame and returns the results.
        return self.model.predict(future)


seasonality = {
    'yearly': True,
    'weekly': True,
    'daily': True
}

def train_predict(df_all_data, df_all_train_index, seasonality_params=seasonality):
    # grab split data
    df_train = df_all_data.copy().iloc[0:df_all_train_index]
    df_test = df_all_data.copy().iloc[df_all_train_index:]

    # To track it on MLflow server. Need to run the server first
    #mlflow.set_tracking_uri("http://127.0.0.1:5000")

    # Useful for multiple runs (only doing one run in this sample notebook)
    with mlflow.start_run():
        
        # Create the Prophet model.
        model = Prophet(
            yearly_seasonality=seasonality_params['yearly'],
            weekly_seasonality=seasonality_params['weekly'],
            daily_seasonality=seasonality_params['daily']
        )
        # Train and predict.
        model.fit(df_train)

        # CV beings with the first 366 days of data to train the model.
        # Predict the next 365 days and evaluates how accurate those predictions are.
        # Shift the window forward by 180 days.
        # Horizon: length of time you want to predict into the future after each training period.
        df_cv = cross_validation(model, initial="366 days", period="180 days", horizon="365 days")
        df_p = performance_metrics(df_cv)

        # Print out metrics
        print("  CV: \n%s" % df_cv.head(2))
        print()
        print("  Perf: \n%s" % df_p.head(2))
        print()

        # Log parameter, metrics, and model to MLflow tracking server, in this case, only RMSE.
        mlflow.log_metric("rmse", df_p.loc[0, "rmse"])

        # Log the trained Prophet model to MLflow
        mlflow.pyfunc.log_model("model", python_model=ProphetWrapper(model))
        
        print("Logged model with URI: runs:/{run_id}/model".format(run_id=mlflow.active_run().info.run_id))

    predicted = model.predict(df_test)
    return predicted, df_train, df_test


if __name__ == "__main__":
    # Read in Data
    df = pd.read_csv('train_data.csv', low_memory=False)
    
    # Data preprocessing
    df = prep_store_data(df)

    # Data splitting of 80% training and 20% test.
    train_index = int(0.8 * df.shape[0])
    
    train_predict(
        df_all_data=df,
        df_all_train_index=train_index,
        seasonality_params=seasonality
    )
