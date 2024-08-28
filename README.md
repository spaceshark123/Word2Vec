

# Java Word2Vec

This is a Java implementation of the popular Word2Vec algorithm for converting words into multidimensional vectors in embedding space. It supports both CBOW (Continuous Bag of Words) and Skip-gram protocols for predicting a word from context words or vice versa, as well as direct vector operations like cosine similarity and embedding vector arithmetic to manipulate words. The included `Word2Vec` class also has several useful helper functions like finding similar words, cleaning/reading from a corpus text, and training the neural network with customizable parameters. It uses a dependency on my java __[NeuralNetwork](https://github.com/spaceshark123/NeuralNetwork)__ project to act as the base for the actual prediction and vectorization process.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Internal Usage](#internal-usage)
- [Program Usage](#program-usage)

## Overview

This project implements Word2Vec, a technique for learning word embeddings using a simple neural network architecture. Word2Vec learns high-dimensional vector representations of words from large text corpora, which capture semantic and syntactic similarities between words. This implementation includes two primary training protocols:

- __CBOW (Continuous Bag of Words)__: Predicts the target word given the surrounding context words.
- __Skip-gram__: Predicts the context words given a single target word.

These embeddings can be used for various natural language processing tasks, such as finding similar words, word analogies, and more.

## Key Features

- __CBOW and Skip-gram Support__: Implements both Continuous Bag of Words and Skip-gram models to allow flexible training and testing.
- __Customizable Parameters__: Allows users to adjust parameters such as learning rate, embedding size, context window size, minimum word frequency, and number of training epochs.
- __Vector Operations__: Supports vector arithmetic and cosine similarity calculations to find relationships between words.
- __Corpus Processing__: Includes functionality to read, clean, and preprocess text corpora, handling tokenization and normalization.
- __Save and Load Models__: Ability to save trained models to a file and load pre-trained models for reuse or evaluation.

## Internal Usage

For use in your own Java projects, simply import the `Word2Vec.java` class file and it will immediately be usable. The following section covers the proper syntax for 

1. __Initializing Word2Vec Model__:

```java
Word2Vec.ModelType modelType = Word2Vec.ModelType.CBOW; // or Word2Vec.ModelType.SKIPGRAM
int minFrequency = 5; // minimum times a word needs to occur for it to be added to the model's vocabulary
int windowSize = 5; // how far around the word to look for context
int dimensions = 100; // how many dimensions should the embedding vector have
String corpusString = "The quick brown fox jumps over the lazy dog."; // corpus text is automatically cleaned up for tokenization and parsing

Word2Vec model = new Word2Vec(modelType, corpusString, minFrequency, windowSize, dimensions);
```

2. __Training the Model__: Use the trainModel method to train the Word2Vec model on a given text corpus.

```java
model.train(numberOfEpochs, learningRate);
```

3. __Finding Similar Words__: Use the findSimilarWords method to find words similar to a given input word or embedding vector.

```java
String[] similarWords = model.findSimilarWords("word", topN);

double[] vector = {...};
String[] similarWords2 = model.findSimilarWords(vector, topN);
```

4. __Vector Arithmetic__: Perform operations like word analogies using vector arithmetic.

```java
double[] kingVector = model.vector("king");
double[] manVector = model.vector("man");
double[] womanVector = model.vector("woman");
double[] queenVector = model.add(model.subtract(kingVector, manVector), womanVector); // king - man + woman = queen
String queenWord = findSimilarWords(queenVector, 1); // gets the closest word that matches this new embedding vector
```

5. __Comparing Words__: Compare the similarity of 2 words using cosine similarity in the similarity() function

```java
double sim1 = model.similarity("king", "queen"); // high similarity (strong correlation)
double sim2 = model.similarity("king", "phone"); // near 0 similarity (no correlation)
double sim3 = model.similarity("king", "peasant"); // low similarity (opposite correlation)
```

## Program Usage

1. **Compile the Code**: First, make sure you are working in the project directory. If you are running the full project with the console interface, run the following commands to compile and run the program:

	***Unix (Mac/Linux) users***:

	__Compile__:

   ```shell
   javac -cp ".:./libraries/jfreechart-1.5.3.jar" Main.java
	```
	__Run__:
	```shell
	java -cp ".:./libraries/jfreechart-1.5.3.jar" Main
	```

	***Windows users***:

	__Compile__:

   ```shell
   javac -cp ".;./libraries/jfreechart-1.5.3.jar" Main.java
	```
	__Run__:
	```shell
	java -cp ".;./libraries/jfreechart-1.5.3.jar" Main
	```

	Or, if you are just using the `Word2Vec` class, the jfreechart library can be excluded, simplifying the commands to:

	__Compile__:

   ```shell
   javac Main.java
	```
	__Run__:

	```shell
	java Main
	```

#### Exiting the Program:

- To exit the program, simply type exit, and the program will terminate.