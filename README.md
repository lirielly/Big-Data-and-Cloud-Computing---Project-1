# Project 1 - Big Data and Cloud Computing

**App Engine Link**: https://imposing-kite-140412.uc.r.appspot.com


## Summary

In this project I have to program an AppEngine app that provides information about images taken from the Open Images dataset. Additionally, the app employs a TensorFlow model for image classification derived using AutoML Vision.

### The primary work will consist in:

+ Defining a BigQuery data set from given CSV files using a Python script or Colab notebook.
+ Programming the app endpoints that are missing. Each endpoint needs to query data stored in BigQuery, and reference the corresponding image files stored in a public storage bucket;
+ Deriving your own TensorFlow model with AutoML, replacing the one that is provided with the application. AutoML requires a dataset that you should define using a Python script or a Colab notebook.

![image](https://user-images.githubusercontent.com/17788854/177190054-e76eab31-4fb6-4351-8786-3f0fe7310819.png)


### Data model
![image](https://user-images.githubusercontent.com/17788854/177189967-d4bfd3aa-0aaf-45aa-9855-c5ad85248072.png)

### Additional challenges
**Use of the Cloud Vision API**
+ Develop an alternative app endpoint for image classification that makes use of label detection through the Google Cloud Vision API using the corresponding Python client API.

**Define a Docker image for the app**
+ AppEngine is enabled by containers internally. Why not define your own container explicitly for the app? All it takes is a DockerFile :)

+ The app should run in the Cloud Shell environment as a container instance, and then also through Cloud Run.
