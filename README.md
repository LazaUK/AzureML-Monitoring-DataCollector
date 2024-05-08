# Inference data collection from Azure ML managed online endpoints

As described in [Azure ML documentation](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-collect-production-data), you can _azureml-ai-monitoring_ Python package to collect real-time inference data received and produced by your machine learning model, deployed to Azure ML managed online endpoint.

This repo provides all the required resources to deploy and test a Data Collector solution end-to-end.

## 1 - Dependency files
Successful deployment depends on the following 3 files, borrowed from the original [Azure ML examples](https://github.com/Azure/azureml-examples/tree/main/sdk/python/endpoints/online/model-1) repo: _inference model_, _environment configuration_ and _scoring script_.

### 1.1 - Inference model
**sklearn_regression_model.pkl** is a SciKit-Learn sample regression model in a pickle format. We'll re-use it "as is".

### 1.2 - Environment configuration
**conda.yaml** is our Conda file, to define running environment for our machine learning model. It has been modified to include the following AzureML monitoring Python package.
``` JSON
azureml-ai-monitoring
```

### 1.3 - Scoring script

## 2 - Solution deployment and testing
