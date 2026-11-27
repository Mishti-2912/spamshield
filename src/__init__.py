"""SpamShield - Spam & Phishing Email Classifier using Naive Bayes.

This package contains the core building blocks of the SpamShield project:

- data_loader   : load and prepare the spam/ham dataset
- preprocessing : clean raw email/message text
- eda           : exploratory data analysis and visualizations
- features      : convert text to numbers using TF-IDF
- model         : train a Multinomial Naive Bayes classifier
- evaluate      : measure model performance (accuracy, precision, recall, F1)
- predict       : classify a single message as Spam or Legitimate
"""

__version__ = "1.0.0"
