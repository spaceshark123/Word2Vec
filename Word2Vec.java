import java.util.*;
import java.io.*;

public class Word2Vec {
    NeuralNetwork net;
    //vocab variable is a hashmap that maps a word to its index in the vocabulary
    HashMap<String, Integer> vocab;
    int numWords;
    String corpus;
    int windowSize;
    int dimensions;
    ModelType modelType;

    List<double[]> inputs = null;
    List<double[]> outputs = null;

    public static enum ModelType {
        CBOW, SKIPGRAM
    }
    
    public Word2Vec(ModelType modelType, String corpus, int minFrequency, int windowSize, int dimensions) {
        this.corpus = cleanText(corpus);
        System.out.println("cleaned corpus: " + this.corpus);
        this.vocab = generateVocab(this.corpus, minFrequency);
        //print the vocabulary
        System.out.println("Vocabulary: " + vocab);
        this.numWords = vocab.size();
        this.windowSize = windowSize;
        this.dimensions = dimensions;
        this.modelType = modelType;

        this.net = new NeuralNetwork(new int[] { numWords, dimensions, numWords },
                new String[] { "linear", "linear", "softmax" });
        //init network with random weights and 0 biases
        this.net.Init(0);
    }

    public NeuralNetwork getNetwork() {
        return net;
    }

    public double[] evaluate(String... words) {
        return net.Evaluate(wordOneHot(words));
    }

    public double accuracy() {
        //use training data to calculate accuracy
        double correct = 0;
        for (int i = 0; i < inputs.size(); i++) {
            double[] output = net.Evaluate(inputs.get(i));
            int maxIndex = maxIndex(output);
            if (maxIndex == maxIndex(outputs.get(i))) {
                correct++;
            }
        }
        return correct / inputs.size();
    }

    public String[] predict(int numPredictions, String... words) {
        //validate input (check if words are in the vocabulary)
        for (String word : words) {
            if (!vocab.containsKey(word)) {
                System.out.println("Word not in vocabulary: " + word);
                return new String[] { "Error" };
            }
        }

        double[] output = net.Evaluate(wordOneHot(words));
        //return the top numPredictions words
        String[] predictions = new String[numPredictions];
        for (int i = 0; i < numPredictions; i++) {
            int maxIndex = maxIndex(output);
            predictions[i] = indexWord(maxIndex);
            output[maxIndex] = 0;
        }
        return predictions;
    }

    // get the probabilities stored in the output layer of the network (after evaluation)
    public double[] getProbabilities() {
        return net.GetNeurons()[net.GetNeurons().length - 1];
    }
    
    //given an embedding vector, find the n closest words in the vocabulary
    public String[] findSimilarWords(double[] vector, int n) {
        //calculate cosine similarity between the vector and all word vectors
        double[] similarities = new double[numWords];
        for (int i = 0; i < numWords; i++) {
            similarities[i] = similarity(vector(indexWord(i)), vector);
        }
        //return the top n words
        String[] closestWords = new String[n];
        for (int i = 0; i < n; i++) {
            int maxIndex = maxIndex(similarities);
            closestWords[i] = indexWord(maxIndex);
            similarities[maxIndex] = 0;
        }
        return closestWords;
    }

    //given an embedding vector, find the n closest words in the vocabulary
    public String[] findSimilarWords(String word, int n) {
        double[] vector = vector(word);
        //calculate cosine similarity between the vector and all word vectors
        double[] similarities = new double[numWords];
        for (int i = 0; i < numWords; i++) {
            similarities[i] = similarity(vector(indexWord(i)), vector);
        }
        //return the top n words
        String[] closestWords = new String[n];
        for (int i = 0; i < n; i++) {
            int maxIndex = maxIndex(similarities);
            closestWords[i] = indexWord(maxIndex);
            similarities[maxIndex] = 0;
        }
        return closestWords;
    }

    // generate training data for the network
    public void generateTrainingData() {
        String[] words = corpus.split(" ");
        inputs = new ArrayList<>();
        outputs = new ArrayList<>();
        for (int i = 0; i < numWords; i++) {
            if (!vocab.containsKey(words[i])) {
                continue;
            }
            String[] contextWords = new String[2 * windowSize];
            int index = 0;
            for (int j = i - windowSize; j <= i + windowSize; j++) {
                if (j != i && j >= 0 && j < numWords && vocab.containsKey(words[j])) {
                    contextWords[index] = words[j];
                    index++;
                }
            }
            //differentiate between CBOW and Skipgram
            if (modelType == ModelType.CBOW) {
                //CBOW
                //input is the context words
                inputs.add(wordOneHot(Arrays.copyOfRange(contextWords, 0, index)));
                //output is the target word
                outputs.add(wordOneHot(words[i]));
            } else {
                //Skipgram
                //create input-output pairs for each context word
                for (int j = 0; j < index; j++) {
                    if (contextWords[j] != null && vocab.containsKey(contextWords[j])) {
                        inputs.add(wordOneHot(words[i]));
                        outputs.add(wordOneHot(contextWords[j]));
                    }
                }
            }
        }
    }

    // train the network
    public void train(int epochs, double learningRate) {
        //generate training data (input and outputs)
        if(inputs == null || outputs == null) {
            generateTrainingData();
        }
        //dummy test data (not used)
        double[][] testInputs = new double[][] {
            wordOneHot(vocab.keySet().toArray(new String[0]))
        };
        double[][] testOutputs = new double[][] {
            wordOneHot(vocab.keySet().toArray(new String[0]))
        };
        System.out.println("Training data size: " + inputs.size());
        //train the network
        NeuralNetwork.TrainingCallback callback = new ChartUpdater(epochs);
        int batchSize = 1;
        net.Train(inputs.toArray(new double[0][]), outputs.toArray(new double[0][]), testInputs, testOutputs, epochs,
                learningRate, batchSize, "categorical_crossentropy", 0,
                new NeuralNetwork.OptimizerType.Adam(0.9, 0.999), callback);
    }

    // one-hot encode a word or multiple words into a vocabulary vector
    public double[] wordOneHot(String... words) {
        //take the average of the one-hot vectors of the words
        double[] oneHot = new double[numWords];
        if (words.length == 1) {
            //one word
            if(words[0] == null) {
                return oneHot;
            }
            oneHot[wordIndex(words[0])] = 1;
            return oneHot;
        }

        //multiple words
        for (String word : words) {
            if(word == null) {
                continue;
            }
            oneHot[wordIndex(word)] = 1.0 / words.length;
        }
        return oneHot;

    }

    //get the embedding vector of a word
    public double[] vector(String word) {
        double[] vector = new double[dimensions];
        for (int i = 0; i < dimensions; i++) {
            vector[i] = net.GetWeights()[1][i][wordIndex(word)];
        }
        return vector;
    }

    // calculate cosine similarity between two words
    public double similarity(String word1, String word2) {
        double[] vector1 = vector(word1);
        double[] vector2 = vector(word2);
        double dotProduct = 0;
        double norm1 = 0;
        double norm2 = 0;
        for (int i = 0; i < dimensions; i++) {
            dotProduct += vector1[i] * vector2[i];
            norm1 += vector1[i] * vector1[i];
            norm2 += vector2[i] * vector2[i];
        }
        norm1 = Math.sqrt(norm1);
        norm2 = Math.sqrt(norm2);
        return dotProduct / (norm1 * norm2);
    }

    // calculate cosine similarity between two words
    public double similarity(double[] vec1, double[] vec2) {
        double dotProduct = 0;
        double norm1 = 0;
        double norm2 = 0;
        for (int i = 0; i < dimensions; i++) {
            dotProduct += vec1[i] * vec2[i];
            norm1 += vec1[i] * vec1[i];
            norm2 += vec2[i] * vec2[i];
        }
        norm1 = Math.sqrt(norm1);
        norm2 = Math.sqrt(norm2);
        return dotProduct / (norm1 * norm2);
    }

    public String cleanText(String text) {
        // Remove punctuation
        text = text.replaceAll("[^a-zA-Z ]", "");
        // Convert to lowercase
        text = text.toLowerCase();
        // remove newlines, tabs, carriage returns
        text = text.replaceAll("[\n\t\r]", " ");
        // remove numbers
        text = text.replaceAll("[0-9]", "");
        // remove stopwords (using nltk's stopwords list)
        String[] stopwords = { "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
                "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its",
                "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
                "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having",
                "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
                "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before",
                "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
                "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
                "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
                "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "also" };
        for (String word : stopwords) {
            text = text.replaceAll("\\b" + word + "\\b", "");
        }
        //trim extra spaces (before and after words)
        text = text.trim();
        //remove extra spaces between words
        text = text.replaceAll(" +", " ");
        return text;
    }

    public HashMap<String, Integer> generateVocab(String text, int minFrequency) {
        HashMap<String, Integer> vocab = new HashMap<>();
        String[] words = text.split(" ");
        for (String word : words) {
            if (vocab.containsKey(word)) {
                vocab.put(word, vocab.get(word) + 1);
            } else {
                vocab.put(word, 1);
            }
        }
        // Remove words that occur less than minFrequency times
        vocab.entrySet().removeIf(entry -> entry.getValue() < minFrequency);
        // Assign indices to words
        int index = 0;
        for (String word : vocab.keySet()) {
            vocab.put(word, index);
            index++;
        }
        return vocab;
    }

    // get the index of a word in the vocabulary
    public int wordIndex(String word) {
        return vocab.get(word);
    }

    // get the word at a given index
    public String indexWord(int index) {
        for (Map.Entry<String, Integer> entry : vocab.entrySet()) {
            if (entry.getValue() == index) {
                return entry.getKey();
            }
        }
        return null;
    }

    // get index of the maximum value in an array
    public int maxIndex(double[] arr) {
        int maxIndex = 0;
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > arr[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    // add two vectors
    public double[] add(double[] arr1, double[] arr2) {
        if (arr1.length != arr2.length) {
            return null;
        }
        double[] sum = new double[arr1.length];
        for (int i = 0; i < arr1.length; i++) {
            sum[i] = arr1[i] + arr2[i];
        }
        return sum;
    }
    
    // subtract two vectors
    public double[] subtract(double[] arr1, double[] arr2) {
        if (arr1.length != arr2.length) {
            return null;
        }
        double[] diff = new double[arr1.length];
        for (int i = 0; i < arr1.length; i++) {
            diff[i] = arr1[i] - arr2[i];
        }
        return diff;
    }
}
