from azureml.ai.monitoring import Collector
from azureml.ai.monitoring.context import BasicCorrelationContext
import pandas as pd

import os
import logging
import json
import numpy
import joblib

def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """

    # --> Data Collector variables
    global inputs_collector, outputs_collector, artificial_context
    inputs_collector = Collector(name='model_inputs')          
    outputs_collector = Collector(name='model_outputs')
    artificial_context = BasicCorrelationContext(id='Laziz_Demo')
    print("========================================")
    print("--Initialised data collector variables--")
    print("========================================")

    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "model/sklearn_regression_model.pkl"
    )
    # deserialize the model file back into a sklearn model
    model = joblib.load(model_path)
    logging.info("Init complete")


def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    """
    logging.info("Request received")
    data = json.loads(raw_data)["data"]
    data = numpy.array(data)

    # --> Data Collector input variable
    input_df = pd.DataFrame(data) # Convert to DF
    context = inputs_collector.collect(input_df , artificial_context) # Correlation context

    result = model.predict(data)

    # --> Data Collector output variable
    output_df = pd.DataFrame(result) # Convert to DF
    outputs_collector.collect(output_df, context) # Outputs data collector

    logging.info("Request processed")
    return result.tolist()
