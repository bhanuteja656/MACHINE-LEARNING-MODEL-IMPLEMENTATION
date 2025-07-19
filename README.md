# MACHINE-LEARNING-MODEL-IMPLEMENTATION

# MACHINE-LEARNING-MODEL-IMPLEMENTATION

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: KOLLURI BHANUTEJA

*INTERN ID*: CT08DF855

*DOMAIN*: PYTHON

*DURATION*: 8 WEEKS

*MENTOR*: NEELA SANTOSH

##In this project, I developed a machine learning model using Scikit-learn to detect whether a given text message is spam or not. The goal was to create a predictive system that classifies messages as spam (1) or ham (0) using natural language processing (NLP) techniques and a classification algorithm.

1. Loading and Exploring the Dataset
I used the SMS Spam Collection Dataset, which contains over 5,000 labeled text messages. Each message is labeled either as "spam" (unwanted messages) or "ham" (legitimate messages). The dataset was loaded directly from a public GitHub link into a pandas DataFrame. I checked the first few rows using .head() and verified the data was structured correctly with two columns: label and message.

2. Preprocessing and Label Encoding
Next, I converted the string labels into numeric format using pandas' .map() function. I encoded ham as 0 and spam as 1, making the data suitable for model training. I also verified the label distribution using .value_counts() to check for class imbalance.

3. Splitting the Dataset
To train and test the model fairly, I split the dataset into a training set (80%) and a testing set (20%) using train_test_split() from Scikit-learn. This step ensures that the model is evaluated on unseen data, giving a true indication of performance.

4. Feature Extraction Using TF-IDF
Since the input data is text, I used the TF-IDF Vectorizer to convert the raw messages into numerical feature vectors. TF-IDF helps by reducing the impact of commonly used words (like “the”, “and”, “is”) and emphasizes words that are more meaningful in the context of spam detection.

5. Model Selection and Training
For classification, I chose the Multinomial Naive Bayes algorithm, which is highly efficient for text classification problems. I trained the model using the training data transformed by the TF-IDF vectorizer. The training process was fast and effective.

6. Testing and Evaluation
After training, I predicted the outcomes on the test set and compared them to the actual labels. I evaluated the model using accuracy, confusion matrix, and the classification report (precision, recall, F1-score). The model achieved high accuracy (around 97–98%), and the confusion matrix showed very few false positives or false negatives.

7. Custom Prediction
To test the model in a real-world scenario, I added a custom message like "Congratulations! You've won a free ticket!" and used the trained vectorizer and model to predict its label. As expected, it was classified as spam, confirming that the model works well on new data.

Conclusion
Overall, I successfully built a spam detection system using Scikit-learn and TF-IDF. The project demonstrated key machine learning steps: preprocessing, feature extraction, model training, evaluation, and prediction. It gave me hands-on experience with text classification and practical use of Python and Scikit-learn libraries.


#OUTPUT

<img width="1440" height="818" alt="Image" src="https://github.com/user-attachments/assets/47d8521d-27b7-4798-80c5-04d20e9215b4" />
