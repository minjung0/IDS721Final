## Final Project

- Build a containerized or PaaS machine learning prediction model and deploy it in a scalable, and elastic platform:
- ML Framework
  - Sklearn, MXNet, PyTorch or Keras/TF
- Model
  - Your own supervised ML prediction model or a Kaggle Prediction Model
- Platform
  - Flask + Kubernetes deployed to EKS (Elastic Kubernetes) or Google Kubernetes Engine
  - Flask + Google App Engine
  - AWS Sagemaker
  - Other (Upon Request)
  
________

### 1. Build a machine learning model using Amazon SageMaker

This is a machine learning model built on Amazon SageMaker, using TensorFlow and Keras for image labeling. The model was trained on the CIFAR-100[^1] dataset, and all the code was developed within the SageMaker Studio environment, which allowed for efficient experimentation and iteration.

[^1]: This dataset is just like the CIFAR-10[^2], except it has 100 classes containing 600 images each. There are 500 training images and 100 testing images per class. The 100 classes in the CIFAR-100 are grouped into 20 superclasses. Each image comes with a "fine" label (the class to which it belongs) and a "coarse" label (the superclass to which it belongs).
[^2]: The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.  
The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class. 

1. Create IAM Role and SageMaker Domain

<img width="700" alt="Screenshot 2023-04-30 at 7 51 34 PM" src="https://user-images.githubusercontent.com/90014065/235381908-827082d0-834f-4843-b57e-b24f2cdabb1e.png">

2. Upload CIFAR-100 train/eval data set to S3 bucket

- Download CIFAR-100 dataset from Keras within SageMaker Studio.
- Save the train/eval set in npy format to your S3 bucket. If the bucket does not exist, it will be created automatically.

<img width="700" alt="Screenshot 2023-04-30 at 7 57 23 PM" src="https://user-images.githubusercontent.com/90014065/235382118-8b5992b6-1e13-4e86-9da6-a3c7801dda19.png">

3. Create and train a machine learning model with Amazon SageMaker

<img width="800" alt="Screenshot 2023-04-30 at 8 12 27 AM" src="https://user-images.githubusercontent.com/90014065/235382208-90d688ec-d1bf-4b57-88ea-70087d249a44.png">

4. Deploy your machine learning model to a SageMaker endpoint

<img width="800" alt="Screenshot 2023-04-30 at 8 01 57 PM" src="https://user-images.githubusercontent.com/90014065/235382256-9c3e4149-528f-47a6-b630-97a859e00f23.png">


### 2. Check the results from the local machine

<img width="138" alt="Screenshot 2023-04-30 at 8 03 00 PM" src="https://user-images.githubusercontent.com/90014065/235382316-806cc3bb-ceeb-4e15-9540-500b2da2e963.png">

<img width="142" alt="Screenshot 2023-04-30 at 8 03 12 PM" src="https://user-images.githubusercontent.com/90014065/235382319-6b54b4a1-cbbf-4b98-ab2d-a6ff62884fa2.png">

<img width="138" alt="Screenshot 2023-04-30 at 8 03 22 PM" src="https://user-images.githubusercontent.com/90014065/235382323-48bc6753-28c8-4a62-a7bc-fd33ecd6e40c.png">

### 3. Check the results from the web using Elastic Beanstalk

1. Create an Elastic Beanstalk application and a configuration

<img width="700" alt="Screenshot 2023-04-30 at 4 48 36 PM" src="https://user-images.githubusercontent.com/90014065/235382367-3125a4e0-2047-41a9-9694-14c141de3f3c.png">

2. Check the results from the web

- When an image is uploaded via the Elastic Beanstalk domain, it is stored in an S3 bucket, and a predicted label is generated using the deployed SageMaker endpoint.

<img width="307" alt="Screenshot 2023-04-30 at 8 05 52 PM" src="https://user-images.githubusercontent.com/90014065/235382507-341ad62a-6e8e-4283-972f-b9558d92e259.png">

