---
layout: post
title: Predicting NASDAQ Composite Index with Daily News
subtitle: Comparitive Study of Multiple Models
tags: [rnn, gru, nlp, embeddings, neural network, nltk]
comments: true
---
In the past years we have seen how recurrent effective  neural networks are in processing and realizing sequential data. Here I want to compare  Machine Learning Models such as Logistic Regression, MLP and RNNs (Word and Character Level) to process daily news and predict the NASDAQ index trends based on daily news articles. And maybe even make money out of it. (Evil laughter...)

### Introduction
Everyone wants to predict the stock market, it has the capacity to make you a lot of money. But the market is influenced by a lot of factors, which makes it hard to predict the movements. Even a little bit of improvement in accuracy can  can yield huge profits in the financial markets. This makes this area even more exciting because  of the risk and the clear monetary benefit. 
One factor, that I choose to look at was the news, which influences the sentiment of the investors. We live in an age where information (or misinformation) travels at such a high pace. I want to see how does it affect the markets and explore if there is any correlation here or not.

With the introduction of RNNs and then LSTMs and GRUs, we can  process sequential data much better. I know more powerful models such as <a href="https://arxiv.org/abs/1706.03762" target="_blank">Transformer</a> exist, but I will talk about it and it's derivatives in a future post. Now, let's focus on the basics here. 
I would like to mention word embeddings or word representational vectors here as well. These vectors help improve the model performance. But as everything is, they come with their issues as well.  
Now, let's try to build a few models and see what models perform the best.... 

### Related Work

As this being such a lucrative topic, I wasn't the first one to think about this at all. This has been done in the past but I wanted to do the evaluations myself. The amount of papers that I could find, was less than what I had hoped for. But this makes sense, corporations investing a lot of money into this don't want to lose their edge by sharing the research. But we still we find a few blog posts trying to do the same and a couple of papers both trying to predict trends of the market using news. 
“Using NLP on news headlines to predict index trends”, Marc Velay and Fabrice Daniel (2018)
“Stock Market Prediction with Deep Learning: A Character-based Neural Language Model for Event-based Trading”, Leonardo d. S. Pinheiro, Mark Dras (2017)

The first paper mentioned above uses a Bag Of Words model, Neural Networks and various other non deep learning approaches such as SVMs, Decision Trees etc to predict the DJIA Index using global news headlines. They found that the bag of words (Logistic Regression models) performs the best out of all the other models. I investigated this claim myself, with a bigger data-set that I made using other data-sets. The second paper listed here, works  with the deep learning approaches such as character level LSTMs. They also use character embeddings in the models, which they trained using a character level language model on the data-set they use. According to this paper, a character level RNN with letter embeddings performs the best with an accuracy of about 60%.

### Description

I'll go through everything I did in order. Also, there's a notebook that contains the code, so you can follow along. Sometimes it's easier to just read the code understand what's happening.

First things first, data collection. I  gathered from two sources:
* "All the News - Components." 18 Jan. 2019. <a href="https://components.one/datasets/all-the-news-articles-dataset/" target="_blank">Source</a>
* "Historical Intraday Nasdaq Composite (COMP) Price ... - FirstRate Data." <a href="http://firstratedata.com/i/index/COMP" target="_blank">Source</a>.

These are two free data-sets that I found online. I want to say this has to be one of the most difficult tasks of a machine learning project. Being used to work with prepared data-sets, this became a challenge harder than I expected. I must say, if you want to spend money you can get real nice data-sets. Scraping data would have required time, which I did not have due to the deadlines for the course. Yes, I did this as a part of my Natural Language Processing course.  

#### Data Pre-processing
Because I had two different data-sets for news and stocks which spanned different time ranges. So, the firs step is to align the data and then match it. I dropped a few news articles here because of the malformed dates, about half a percent maybe. Also, market data was at the frequency of a minute. And then calculating, the positive or negative change in the closing prices was a little tricky but nothing too elaborate. what I described here lies in some python scripts in the repo, I promise everything else is in the notebook. 

#### Text Pre-processing

Once we have the data ready we can start preparing our text pre-processing pipeline. The pipeline consists of three stages, normalizing the data i.e. removing the punctuation and making all the letters to lowercase. The major reason for lowercase is the embedding matrix. Which contains embeddings corresponding to words in lowercase. More on this later. Then we remove all the stop words which don’t add too much meaning to the sentence. Keeping them might be useful in some cases but we drop them in our case. Once this is done, we turn to the most fun part, Lemmatization i.e. reducing the words into their root forms from their inflected forms. We avoid stemming here, even though it’s way faster but it doesn’t work so well when using embeddings. For example, 'features' turns into 'featur'  which isn’t a word and no embedding exists for it. 

I used NLTK to get the correct part of speech tag and then lemmatize the word. It improves the lemmatization accuracy, by default lemmatizers consider every word to be a noun. I started by doing this using Spacy, but even after disabling a lot of spacy features, it was still slow. Which is why I switched to using NLTK. All this pre-processing is done in parallel using python’s in built multiprocessing module. I divided the data frame and use a pool to process text in parallel which saved me a lot of time. Needless to say, that we use Pandas, Numpy and PyTorch throughout the notebook. Now, we drop the rows which have articles of length more than Torchtext can read in, we lose less than 0.01% of the data by doing this, so we just drop those outliers. 
Also,  I split the data into Train, dev and validation sets. This makes it easier to ingest data into TorchText(Library to help with NLP in PyTorch). Also, store the data onto disk, so that we don’t have to pre-process the data again and again. This saves a lot of time. There are pre-processed samples of processed data in the {ProcessedData} folder in the repository to test the code. 

Once the data is pre-processed and loaded into Torchtext we can start talking about the Models.

### Models ( and helpers )

We work on four different models, we’ll go through each of them, one at a time. Every model will have three major components, the data loader, the model itself and the training module. 

#### Bag of Words Model

The first model is the most basic of them all. 
The data loader for this model turns each article and turns it into a vector of length of the vocabulary. The vector contains the number of times the words appeared in the article. Torchtext, helps up by creating a mapping from index to word and vice versa.
For example, if  our vocabulary was [W1, W2, W3, W4, W5] and the sentence was "W1 W2 W3 W1 W4 WX". Here WX is a unknown word. The input vector would be [1 0 2 1 1 1 0]. Now you might be wondering, why is the vector length is vocab size + 2. The extra two indices are for unknown words and padding. Padding is used to make the sentence of equal lengths, so we can process them in batches. 
We use this long vector and pass it through a single layer network (Logistic Regression) that outputs two nodes. One for positive and other for negative movement. 

The training for this model is done via the training module written for this model which trains in batches and also runs on GPUs if available. Otherwise PyTorch will just run the training on as many cores available. Also, we use CrossEntropy Loss and Adam as the optimizer with default parameters. There are no dropouts in this model but we do use early-stopping to prevent over-fitting, we keep the model which performs best on the validation set.       

This model takes the longest to train even being the simplest of all the models because of the huge size of the input vectors. The vocabulary size was 70K.

##### Training Module

I'll take a segue here to talk about the Training Module because it is shared by the rest of the models. This module contains functions for training and evaluating models. It takes in the model object to be trained, when initializing the module. And then we call  the train_model function which takes in the training and validation set iterators. Early stopping is also implemented here, I deepcopy the best performing model on the validation (or dev) set. The function in the end returns the model which performed the best on the validation set, which we then use to evaluate the model on the test-set.

##### Glove Loader

Continuing the segue, we have a few functions that help us load Glove Embeddings. Create the word to index and index to word mappings and the Embedding Matrix, which can be fed to PyTorch.  The embedding layer in PyTorch can taken in a pre-built embedding matrix.

#### Neural Network Model

This model uses a different data loader which clips the sequences to a max length of 1200. We only clip 0.3% of the data when doing this. And each number in the vector is the word corresponding to that index in the dictionary. But the main difference here is the introduction of word embeddings (Glove). I averaged the embeddings for all the words in the sequence which form the input vector of size same as the embedding dimension. The neural network has two hidden layers and one output layer which are all fully connected. We just use one output here because of binary classification. We again use CrossEntropy Loss but PyTorch has a separate cross entropy loss for the binary cases called Binary Cross Entropy Loss (BCE Loss). We use the one with Logits (BCE Loss with Logits).  It has an extra sigmoid activation included i.e. it applies it to the outputs and we don’t have to do it. The optimizer used is Adam again with the recommended defaults.  

This model is fastest to train and when running on GPU it really shines. We get very good accuracy with this model, I'll put all the results in the together. 

#### Word Level RNN Model

This model uses the same data loader as the previous model. But this model is conceptually very different from the previous model. Here we preserve the sequence nature of the text. I use a GRU Model to summarize the entire sequence of text into a vector. This vector is then passed into the fully connected linear layer to get the result. At each time step of the GRUs, we feed in the embedding for the word. This is similar to Seq2Seq models, where we create a context for the given text. Here I use that context to predict if the stock shows a positive moment or a negative one. 
Again, I used the BCE Loss with Logits again and Adam as the optimizer. This model is second in the training speeds. Slower than the Neural Network but still faster than others. 

#### Character Level RNN Model

For this model we have to re-write the data loader because we work with characters instead of words. So the sequence length is capped at 5600 characters. Also I didn't remove the punctuation or case normalize the data, because we want as much diversity as much as possible because our vocabulary is very limited consisting of numbers, letters or both cases and printable punctuation. 

We do something similar as we did in our previous model, such that we summarize all the sequence into the hidden vector and then pass that vector to a fully connected layer which then produces the output for us. 
We use BCELoss with Logits again here as well as Adam optimizer.  
This model comes second last in training speeds, again because of it being a RNN and a big vector size. 

### Results

This is the fun part, how did it all go. We used the vanilla forms of the models. Let's see how it goes. 

| MODEL               | MAX TRAIN ACCURACY | MAX DEV ACCURACY | TEST ACCURACY |
|---------------------|--------------------|------------------|---------------|
| Bag of Words        | 90.93              | 57.02            | 56.78         |
| Neural Network      | 79.07              | 59.14            | 60.83         |
| RNN Word Level      | 60.77              | 59.07            | 60.79         |
| RNN Character Level | 59.75              | 59.16            | 60.30         |

The results we got for the Bag of Words Model and the RNN Character Level were in line with the related material I mentioned above. But the RNN Word Level and Neural Network Model outperform the other models which I wanted to see. A lot of sources mention that predicting stock markets reliably is nearly impossible. Because there are too many factors involved. But we still see an accuracy of 50%+ which is not by random chance, it is fair to say that there exists some correlation here.
With that in mind these results are respectable but we have to take them with a grain of salt, because the quality of the data wasn’t so good. And I trained on news which came after the markets closed  which isn’t good. Because we added future information, which might have influenced the results. Also, the the ratio of positive to negative movements wasn't 50%. It was about 53.5% positive and 46.5% negative. 

### Analysis 

The Bag of Words model, performs the worst because it just looks at the word and doesn’t impart any meaning to it. Also, just the length of the sequence can change the results, because of aggregation. 
The Neural Network model, performs the best with embeddings. I think it's because of the word embeddings which impart some relationship among the words. Also multiple non-linearities helps approximate the true function. 
The Word-RNN model works really as well. The summarizing of the text is done really well by the RNN which when fed to a single linear layer yields good results.  Comparable to the multi layer perceptron described above. 
The final model with Character Embeddings and a RNN(GRU) works unexpectedly well. I think it is because it can capture the numbers like $100 or 2.69% better than any other which is very frequent in the financial news. This kind of capturing helps summarize the data well. And hence, produces good results. 
Onc more thing we should notice here, is that the Bag of Words and Neural Network model overfit a lot where as the RNN models don't. This shows us that incorporating the sequence nature of the text, does help generalize the models better. 

### Future Work
Now, that we have seen the models does infact gets the predictions right more than 50% of the time. I would like to fine tune it for specific stocks and train the model on the news articles that contain information about those stocks. With having predictions about individual stock is much more useful than predicting the index as a whole. Also, one big improvement that can be made is utilizing the time dependent nature of the market data. Right now the model looks at each day as independent of all the previous day which is definitely not true in the case of market data. We want the model to make a prediction using the current news and market data from the past. And making it multi-class by adding classes according to the percentage change in the market. This would make the model even more reliable and probably give us at least marginally better results.  

### Thoughts
The effort increased as time grew through the project but the stress was the maximum in the start. Collecting the data was the hardest part in the project. And then cleaning the data and then making the model work. I was comfortable with Numpy and Pandas but not so much with PyTorch, which I am now. I spent a lot of time trying out different parameters. It wasn’t so much of fine tuning but more of curiosity to see what change will this parameter cause. 

We all learn better by applying what we learned and this project was no different. I can confidently say that this project helped me become way more confident about writing deep learning models and wrangling the data around.

Let me know what you all think about this in the comments below (). Also, how to make this better. This is my first technical post and I hope to make them better over time. Feedback is appreciated.

Until next time. Cheers !