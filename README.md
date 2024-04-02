# SMS-SPAM-Detection
Machine Learning Project for Text Classification

# 1.	INTRODUCTION
In today’s world there is a big need of analyzing the data as it has lot of information that goes un-noticed. In order to detect such pattern, the Machine Learning techniques are used.
In this project, we are going to work with a dataset which has the data of SMS which are collected to classify them into SPAM or HAM. So, if a message is sent by a real user then it should be tagged as Ham or if it is by a machine for advertisement purpose then it should be tagged as SPAM.
# 1.1	PURPOSE
In daily life, we receive lot of messages from different sources and those messages might be arriving from a machine just for the advertisement purpose. The mobile user gets irritated because of such messages and may even ignore a real message from a known person. In order to avoid such cases, we are going to classify these SMS messages into two categories as SPAM or HAM.
# 1.2	SCOPE
The project is made on Anaconda Jupyter Notebook. This application is an easy to use application as the user need to run the complete program and the user will get the accuracy of each model. 
Since, there is no un-labelled data, we are going to divide the data in Train and Test dataset. Test dataset is used for validating our model performance.
Also, in case in future we have some new dataset which is not labelled then we can classify them too. 
# 1.3	OVERVIEW
This document will give you an easy walk over through the application and act as a guide with easy steps to use and maintain the application. Detailed overview of each feature and design is covered below in the System Overview. This application does not involve any database, but this is a future aspect of this application in case there is a need to store the labelled data, the flow of application is explained with data flow in use case diagram under System Architecture section. In the end, a visual look and feel of this application with the flow of application is shown. This application act as a perfect medium to classify the SMS messages with text data using NLP techniques. 

# 2.	SYSTEM OVERVIEW
The project involves the text data which cannot be classified by basic modeling techniques and hence we are using NLP techniques. Natural Language Processing helps us to bridge the gap of text data with numerical data which is needed to run by the machine.
It also helps us to neutralize the dirty text data into a simpler form. We are removing Stop words punctuations, commonly appearing terms using TF-iDF (Term Frequency inverse Document Frequency).
The major steps that are involved in order to classify the data are as follows. Each step is described with Code Snippet.
# Importing Libraries:
First thing we are going to do is to import the important libraries. Since we are working on Text data, we have imported NLTK for text analysis. Also, for data visualization, we have used matplotlib library.
# Read Dataset: 
Next thing we need to do is to input the dataset we need to work on. The data is located in our current working directory which is Downloads in this case. The command dataframe.head() help us in checking the first 5 lines of our dataframe.
 
# Remove NA values:
The biggest problem in any dataset is the NA values which needs to be handled very carefully. Since they are not going to add any meaning to our classification, we need to remove all the NA fields.
 
# Describe the dataset and the label column:
The data should be understood correctly and hence, we are going to describe the whole dataset and also the label column.
 
Figure 4: Describing the Dataset and the Label Set
This image clearly shows that our data has only two different kind of labels and also the count of unique values.
# For Visualization – Adding new column:
In order to visualize our dataset, we need to add one column which shows the total number of words in that particular feature field.
 
# Basic Visualization of dataset:
Let’s do a simple visualization of data before we start with machine learning.
 
# Machine Learning Steps – Basic Text pre-processing steps:
This step is our first important step and we are first going to do some text pre-processing in order to clean the dataset before we input it to our models.

So, we have converted all the text into lower case and then we removed the punctuations from our dataset. Finally, we removed the stop words form our dataset.

# Stemming each term using Porter Stemmer:
In order to make sure that our model predict the data better, we are going to stem our words into its stem form.

# Split the dataset into train and test:
Since, we do not have any train and test dataset split already, we are going to split the dataset into train and test. The test dataset is used for validating our models.

# Training Models Naïve Bayes:
Let’s start training our models. The first model we are going to use is Naïve Bayes.
In the last line, we have also shown the predicted values.

# Decision Tree:
Similarly, we are going to train our data with Decision Tree.
 
# Random Forest:
Also, we are going to use Random Forest as generally it gives higher accuracy then other models.
 
# Support Vector Machine:
SVMs are always considered as good model for text classification. So, let’s train the SVM too.
 
# Classification Report for all the models:
As we are done with all the models we wanted to use, let’s check the classification report of these models.
 
# Finally Checking the Accuracy Score:
Finally let’s check their accuracy score in order to check which model performed best amongst all.
So, we can see that SVM and Naïve Bayes has given the best accuracy although there is no much difference with other models.

This is a simple implementation of Machine Learning to classify the text data. The code is attached below for more details.
