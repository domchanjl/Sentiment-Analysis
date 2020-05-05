# Sentiment-Analysis
Subreddits' Depression and SuicideWatch

# DSI 13 Individual Submissions
## Project 3 - Web APIs & Classification
---
### Problem Statement

**How can we use predictive modeling to best predict which subreddit a post came from?**

Determining the essential coefficients and significance of words to accurately place a post as belonging to the Subreddits of either "Depression" or "SuicideWatch"; to effectively predict the suicidal posts in the "Depression" subreddit.
This is to enable people in professions charged with the care of others, but do not have the expertise, to identify their mental states – more crucially those negative thoughts tending to suicide. 

Upon reading the posts, it is clear that there are many similar word choices between the 2 subreddits. Given the overlapping nature of both subreddits, it was also crucial that participants of both communities have a better idea of either subreddits so that content can be posted more accurately.

With the anonymity allowed by the advent of the internet, one’s username/handle gives an avenue for people to be someone theyd normally hide irl. They may in fact be more truthful as a result. Reddit was selected for data scraping to explore the virtual suicide/depression notes of people who might be afflicted with either states of mental health. This data could provide the various outreaches with information pertaining to people susceptible to the sweet allure of death

A training dataset was scraped from Reddit, with the inclusion of the target variable as the posts from "SuicideWatch" (is_sw). The challenge was generating a classification model using the training data, to predict post of the target feature. 

---
### Executive Summary

- Data Scraping
- Pre-Processing
- EDA & Text Extraction Feature
- Modelling
- Optimization
- Final Analysis

---
### My Process

#### Data Scraping

Loop to pull Reddit API posts
To collect Reddit data, the existing .json API formatwas used. This returned a dictionary for containing a .json extension. A headers dictionary was included to access Reddit, which allowed to execute an API loop to accumulate the maximum allowed posts (~1000 per subreddit). To prevent Reddit from percieving the extraction fo data as a hack, time.sleep function also introduced. Using nested list comprehensions, for each of the 40 subreddit pages of 25 posts per page, iterate through each individual post dictionary to collect the key attributes of all posts (e.g. title,selftext,etc.). I identified duplicate posts and titles, noting that titles may be duplicated with different posts 

A DataFrame table containing titles, post, upvotes, downvotes, authors, and combined values, as well as our target vector is_sw was then saved.

---
#### Preprocessing & EDA

To deal with text data in the raw posts, the following processes will be employed:
- Tokenizing
- Lemmatizing & Stemming 

The functions should receive one string of text and return the processed text.

Prior to combining the data frames, initial cleaning steps were taken to remove duplicate posts. Following which, a total of 1909 posts remained in total, with 983 posts from r/depression, and 926 posts from r/SuicideWatch. These posts contain very few HTML code artifacts. These functions serves to  remove punctuation, remove stopwords, stem & lemmatize each word of the text. 

Using stemmer to improve the modeling ability of strings which trims characters from each word to convert it to a stem. Similar words will register as equivalent during feature extraction if they share a stem, returning a base form of the word (i.e., computer/computing/computed all result in a stem 'comput'). It tends to be cruder than using lemmatization.

Lemmatizing is a precise way of handling things from a grammatical/morphological point of view. The words attempt to return their lemma, or the base/dictionary form of a word. Lemmatizing was the better function as it was less harsh and did not result in possibly altering the meaning of words.

The effectiveness of text feature selection models when used with the pre-processed data were also assessed. Using the columns of lemmatized words, the two vectorization transformers, CountVectorizer and TF-IDF, are modelled using Naive Bayes, K-Nearest Neighbors, Logistic Regression Classifier. A GridSearch is run across all models to rule out non-viable options. The text feature selection that give the models with the most predictive potential are then selected and optimized in the next notebook.

As the feature words are still unstructured for analysis, employ count vectorization and TF-IDF to transform the lists of the cleaned reviews above into features passable into a model. It will create columns (also knon as vectors), where each column counts how many times each word is observed in each review.

The difference of accuracy scores between training and test set, and test set and baseline accuracy, were measured. This will tell us about the level of overfitting that may be present in each model. The baseline accuracy is the likelihood of a post being is_sw=1 based solely on the percentage of the dataset that is the target value. Normalizing the value counts, and identify majority group and take that as the baseline accuracy of 51.4%.

Assess which models will be the best to optimize by consolidating and sorting the results values by test_accuracy. It is observed that CountVectorized and TF-IDF models performed relatively similarly. Since CountVectorized models registered the highest score, we will use that as our vectorizer.

---
#### Modeling & optimization

Using the post count vectorized parameter above to furthe optimize the models. The GridSearch is set to a random_state value, so that cross validation selection will be consistent between runs, and we will be able to make direct comparisons over effectiveness of hyperparameters.

We start with a wide range for fields of interest, and narrow around the optimally selected value and gauge the degree of accuracy increase (or decrease). Through trial and error, we are able to select hyperparameters that will promote the most accurate modeling results.

---
#### Evaluation
Test accuracy scores for Logistic Regression model was better than K-Nearest Neighbor model, although both were improved with tuning of hyperparameters. Multinomial Naive Bayes, however, saw a decrease in test scores.

Highest-performing model is Logistic Regression, and coefficients allow us to understand the data easily. Moreover, we see improvements to the metrics of accuracy, specificity, sensitivity and precision. Despite tuning the parameters, I could not get better than 69.2% accuracy, unfortunately.

I spent some time trying to generalize pipelines in order to efficiently run lots of different grid searches on many different models and parameters. This generalized function has room for improvement, but was quite helpful for me to stay organized in my tests.




```python

```
