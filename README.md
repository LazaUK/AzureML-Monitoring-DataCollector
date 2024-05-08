# Inference data collection from Azure ML managed online endpoints

As described in [Azure ML documentation](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-collect-production-data), you can _azureml-ai-monitoring_ Python package to collect real-time inference data received and produced by your machine learning model, deployed to Azure ML managed online endpoint.

This repo provides all the required resources to deploy and test a Data Collector solution end-to-end.

## 1 - Dependency files
Successful deployment depends on the following 3 files, borrowed from the original [Azure ML examples](https://github.com/Azure/azureml-examples/tree/main/sdk/python/endpoints/online/model-1) repo: _inference model_, _environment configuration_ and _scoring script_.

### 1.1 - Inference model
**_sklearn_regression_model.pkl_** is a SciKit-Learn sample regression model in a pickle format. We'll re-use it "as is".

### 1.2 - Environment configuration
**_conda.yaml_** is our Conda file, to define running environment for our machine learning model. It has been modified to include the following AzureML monitoring Python package.
``` JSON
azureml-ai-monitoring
```

### 1.3 - Scoring script
**_score_datacollector.py_** is a Python script, used by the managed online endpoint to feed and retrieve data from our inference model. This script was updated to enable data collection operations.

_Collector_ and _BasicCorrelationContext_ classes are referenced, along with the _pandas_ package. Inclusion of pandas is crucial, as Data Collector at the time of writing was able to log directly only DataFrames.
``` Python
from azureml.ai.monitoring import Collector
from azureml.ai.monitoring.context import BasicCorrelationContext
import pandas as pd
```

_init_ function initialises global Data Collector variables.
``` Python
global inputs_collector, outputs_collector, artificial_context
inputs_collector = Collector(name='model_inputs')          
outputs_collector = Collector(name='model_outputs')
artificial_context = BasicCorrelationContext(id='Laziz_Demo')
```

"model_inputs" and "model_outputs" are reserved Data Collector names, used to auto-register relevant Azure ML data assets.
![Screenshot_1.3a](images/screenshot_1_3a.png)


## 2 - Solution deployment and testing
