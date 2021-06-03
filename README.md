# Assignment-5 
The goal is to build a sentiment analysis model using StanfordSentimentAnalysis Dataset.

* Download the StanfordSentimentAnalysis Dataset from this link (http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip) Use "datasetSentences.txt" and "sentiment_labels.txt" files from the zip you just downloaded as your dataset. This dataset contains just over 10,000 pieces of Stanford data from HTML files of Rotten Tomatoes. * * The sentiments are rated between 1 and 25, where one is the most negative and 25 is the most positive.
* Train your model and try and achieve 60%+ accuracy. Upload your colab file on git with the the training logs
![image](https://user-images.githubusercontent.com/10822997/120690224-91939580-c4c2-11eb-9614-02bb4b35e03b.png)

## Dataset
This file includes:

* original_rt_snippets.txt contains 10,605 processed snippets from the original pool of Rotten Tomatoes HTML files. Please note that some snippet may contain multiple sentences.
* dictionary.txt contains all phrases and their IDs, separated by a vertical line |
* sentiment_labels.txt contains all phrase ids and the corresponding sentiment labels, separated by a vertical line. Note that you can recover the 5 classes by mapping the positivity probability using the following cut-offs: [0, 0.2], (0.2, 0.4], (0.4, 0.6], (0.6, 0.8], (0.8, 1.0] for very negative, negative, neutral, positive, very positive, respectively. Please note that phrase ids and sentence ids are not the same.
* SOStr.txt and STree.txt encode the structure of the parse trees. STree encodes the trees in a parent pointer format. Each line corresponds to each sentence in the datasetSentences.txt file. The Matlab code of this paper will show you how to read this format if you are not familiar with it.

* datasetSentences.txt contains the sentence index, followed by the sentence string separated by a tab. These are the sentences of the train/dev/test sets.

* datasetSplit.txt contains the sentence index (corresponding to the index in datasetSentences.txt file) followed by the set label separated by a comma: 1 = train 2 = test 3 = dev

## Model
![image](https://user-images.githubusercontent.com/10822997/120690618-154d8200-c4c3-11eb-87d1-bfb9dbf5961b.png)

## Data Augmentation

Augmented the data using

* ### Random swap - The random swap augmentation takes a sentence and then swaps words within it n times, with each iteration working on the previously swapped sentence.
![image](https://user-images.githubusercontent.com/10822997/120690845-5a71b400-c4c3-11eb-85a4-8968c9d734c1.png)

* ### Random delete - As the name suggests, random deletion deletes words from a sentence. Given a probability parameter p, it will go through the sentence and decide whether to delete a word or not based on that random probability.

![image](https://user-images.githubusercontent.com/10822997/120690981-82f9ae00-c4c3-11eb-93b0-ab6ff298be2a.png)

* ### Back translation using google translate - This involves translating a sentence from our target language into one or more other languages and then translating all of them back to the original language.
![image](https://user-images.githubusercontent.com/10822997/120691063-a02e7c80-c4c3-11eb-847c-3243aaa58f5b.png)
