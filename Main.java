import java.io.*;
import java.util.*;

public class Main {
    static Word2Vec model = null;
    public static void main(String[] args) throws Exception {
        ConsoleTool console = new ConsoleTool(System.in, System.out, "Word2Vec Console\n-=-=-=-=-=-=-=-=-=-=-=-=-=-=-");
        console.addCommand("clear", new ConsoleTool.Command() {
            public void execute(String... arguments) {
                console.Clear();
            }
        });
        console.addCommand("exit", new ConsoleTool.Command() {
            public void execute(String... arguments) {
                console.Output("Exiting...");
                console.finish();
                System.exit(0);
            }
        });
        console.addCommand("create", new ConsoleTool.Command() {
            public void execute(String... arguments) {
                //validate arguments
                if (arguments.length != 5) {
                    console.Output("Usage: create <modelType> <corpus> <minFrequency> <windowSize> <dimensions>");
                    return;
                }
                if (!arguments[0].equals("CBOW") && !arguments[0].equals("SKIPGRAM")) {
                    console.Output("Invalid model type. Choose either CBOW or SKIPGRAM");
                    return;
                }
                File file = new File(arguments[1]);
                if (!file.exists()) {
                    console.Output("File not found");
                    return;
                }
                Word2Vec.progressBar(15, "Initializing Word2Vec model ", 0, 4, "parsing corpus text...");
                String corpus = "";
                try {
                    BufferedReader bf = new BufferedReader(new FileReader(file));
                    String line;
                    while ((line = bf.readLine()) != null) {
                        corpus += line + " ";
                    }
                    bf.close();
                } catch (Exception e) {
                    console.Output("Error reading file");
                    return;
                }
                if(corpus.length() == 0) {
                    console.Output("Empty file");
                    return;
                }
                if (Integer.parseInt(arguments[2]) < 1) {
                    console.Output("Minimum word frequency must at least 1");
                    return;
                }
                if (Integer.parseInt(arguments[3]) < 1) {
                    console.Output("Window size must at least 1");
                    return;
                }
                if (Integer.parseInt(arguments[4]) < 1) {
                    console.Output("Dimensions must at least 1");
                    return;
                }

                Word2Vec.ModelType modelType = Word2Vec.ModelType.valueOf(arguments[0]);
                model = new Word2Vec(modelType, corpus, Integer.parseInt(arguments[2]), Integer.parseInt(arguments[3]), Integer.parseInt(arguments[4]));
                console.Output("Model created");
            }
        });
        console.addCommand("train", new ConsoleTool.Command() {
            public void execute(String... arguments) {
                //validate arguments
                if (arguments.length != 2) {
                    console.Output("Usage: train <epochs> <learningRate>");
                    return;
                }
                if (model == null) {
                    console.Output("Model not created");
                    return;
                }
                if (Integer.parseInt(arguments[0]) < 1) {
                    console.Output("Epochs must at least 1");
                    return;
                }
                if (Double.parseDouble(arguments[1]) <= 0) {
                    console.Output("Learning rate must be greater than 0");
                    return;
                }

                model.train(Integer.parseInt(arguments[0]), Double.parseDouble(arguments[1]));
                console.Output("Model trained");
            }
        });
        console.addCommand("accuracy", new ConsoleTool.Command() {
            public void execute(String... arguments) {
                if (model == null) {
                    console.Output("Model not created");
                    return;
                }
                console.Output("Accuracy: " + model.accuracy() * 100 + "%");
            }
        });
        console.addCommand("save", new ConsoleTool.Command() {
            public void execute(String... arguments) {
                if (model == null) {
                    console.Output("Model not created");
                    return;
                }
                if (arguments.length != 1) {
                    console.Output("Usage: save <filename>");
                    return;
                }
                model.Save(arguments[0]);
                console.Output("Model saved");
            }
        });
        console.addCommand("load", new ConsoleTool.Command() {
            public void execute(String... arguments) {
                if (arguments.length != 1) {
                    console.Output("Usage: load <filename>");
                    return;
                }
                model = Word2Vec.Load(arguments[0]);
                if (model == null) {
                    console.Output("Error loading model");
                    return;
                }
                console.Output("Model loaded");
            }
        });
        console.addCommand("predict", new ConsoleTool.Command() {
            public void execute(String... arguments) {
                if (model == null) {
                    console.Output("Model not created");
                    return;
                }
                if (arguments.length < 1) {
                    console.Output("Usage: predict <words separated by spaces>");
                    return;
                }
                String[] words = model.cleanText(String.join(" ", arguments)).split(" ");
                //validate words
                for(String word : words) {
                    if(!model.isWord(word)) {
                        console.Output("Word not found in vocabulary: " + word);
                        return;
                    }
                }
                String[] predictions = model.predict(5, words);
                if (predictions[0].equals("Error")) {
                    console.Output("Error predicting words");
                    return;
                }
                console.Output("Predictions:");
                for (String prediction : predictions) {
                    int index = model.wordIndex(prediction);
                    double probability = model.getProbabilities()[index];
                    probability = Math.round(probability * 10000.0) / 100.0;
                    console.Output(prediction + " - " + probability + "%");
                }
            }
        });
        console.addCommand("vocab", new ConsoleTool.Command() {
            public void execute(String... arguments) {
                if (model == null) {
                    console.Output("Model not created");
                    return;
                }
                console.Output("Vocabulary:");
                String[] vocabulary = model.getVocabulary();
                console.Output(vocabulary);
                console.Output("Total words: " + vocabulary.length);
            }
        });
        console.addCommand("similarity", new ConsoleTool.Command() {
            public void execute(String... arguments) {
                if (model == null) {
                    console.Output("Model not created");
                    return;
                }
                if (arguments.length != 2) {
                    console.Output("Usage: similarity <word1> <word2>");
                    return;
                }
                if (!model.isWord(arguments[0])) {
                    console.Output("Word not found in vocabulary: " + arguments[0]);
                    return;
                }
                if (!model.isWord(arguments[1])) {
                    console.Output("Word not found in vocabulary: " + arguments[1]);
                    return;
                }
                double similarity = model.similarity(arguments[0], arguments[1]);
                console.Output("Similarity: " + similarity);
            }
        });
        console.addCommand("vector", new ConsoleTool.Command() {
            public void execute(String... arguments) {
                if (model == null) {
                    console.Output("Model not created");
                    return;
                }
                if (arguments.length != 1) {
                    console.Output("Usage: vector <word>");
                    return;
                }
                if (!model.isWord(arguments[0])) {
                    console.Output("Word not found in vocabulary: " + arguments[0]);
                    return;
                }
                double[] vector = model.vector(arguments[0]);
                console.Output(vector);
            }
        });
        //command to get the 5 most similar words to a given word (not called mostSimilar or similar)
        console.addCommand("findsimilar", new ConsoleTool.Command() {
            public void execute(String... arguments) {
                if (model == null) {
                    console.Output("Model not created");
                    return;
                }
                if (arguments.length != 1) {
                    console.Output("Usage: findsimilar <word>");
                    return;
                }
                if (!model.isWord(arguments[0])) {
                    console.Output("Word not found in vocabulary: " + arguments[0]);
                    return;
                }
                String[] similar = model.findSimilarWords(arguments[0], 5);
                console.Output("Similar words:");
                console.Output(similar);
            }
        });
        //command to add 2 words and get the closest word to the result
        console.addCommand("add", new ConsoleTool.Command() {
            public void execute(String... arguments) {
                if (model == null) {
                    console.Output("Model not created");
                    return;
                }
                if (arguments.length != 2) {
                    console.Output("Usage: add <word1> <word2>");
                    return;
                }
                if (!model.isWord(arguments[0])) {
                    console.Output("Word not found in vocabulary: " + arguments[0]);
                    return;
                }
                if (!model.isWord(arguments[1])) {
                    console.Output("Word not found in vocabulary: " + arguments[1]);
                    return;
                }
                double[] vector1 = model.vector(arguments[0]);
                double[] vector2 = model.vector(arguments[1]);
                double[] result = model.add(vector1, vector2);
                console.Output(model.getClosestWord(result));
            }
        });
        //command to subtract 2 words and get the closest word to the result
        console.addCommand("subtract", new ConsoleTool.Command() {
            public void execute(String... arguments) {
                if (model == null) {
                    console.Output("Model not created");
                    return;
                }
                if (arguments.length != 2) {
                    console.Output("Usage: subtract <word1> <word2>");
                    return;
                }
                if (!model.isWord(arguments[0])) {
                    console.Output("Word not found in vocabulary: " + arguments[0]);
                    return;
                }
                if (!model.isWord(arguments[1])) {
                    console.Output("Word not found in vocabulary: " + arguments[1]);
                    return;
                }
                double[] vector1 = model.vector(arguments[0]);
                double[] vector2 = model.vector(arguments[1]);
                double[] result = model.subtract(vector1, vector2);
                console.Output(model.getClosestWord(result));
            }
        });
        console.addCommand("help", new ConsoleTool.Command() {
            public void execute(String... arguments) {
                // if no arguments are provided, print general help, otherwise print help for the specific command
                if (arguments.length == 0) {
                    console.Output("Commands:");
                    console.Output("create <modelType> <corpus> <minFrequency> <windowSize> <dimensions> - Create a new Word2Vec model");
                    console.Output("train <epochs> <learningRate> - Train the model");
                    console.Output("accuracy - Display the accuracy of the model");
                    console.Output("save <filename> - Save the model to a file");
                    console.Output("load <filename> - Load the model from a file");
                    console.Output("predict <words> - Predict the next words");
                    console.Output("vocab - Display the vocabulary of the model");
                    console.Output("similarity <word1> <word2> - Display the cosine similarity between two words");
                    console.Output("vector <word> - Display the embedding vector of a word");
                    console.Output("findsimilar <word> - Display the 5 most similar words to a word");
                    console.Output("add <word1> <word2> - Add two words and display the closest word to the result");
                    console.Output("subtract <word1> <word2> - Subtract two words and display the closest word to the result");
                    console.Output("clear - Clear the console");
                    console.Output("exit - Exit the console");
                } else {
                    String command = arguments[0];
                    switch (command) {
                        case "create":
                            console.Output(
                                    "create <modelType> <corpus> <minFrequency> <windowSize> <dimensions> - Create a new Word2Vec model");
                            console.Output("modelType: CBOW or SKIPGRAM");
                            console.Output("corpus: Path to the corpus file");
                            console.Output("minFrequency: Minimum word frequency");
                            console.Output("windowSize: Window size");
                            console.Output("dimensions: Number of dimensions");
                            break;
                        case "train":
                            console.Output("train <epochs> <learningRate> - Train the model");
                            console.Output("epochs: Number of epochs");
                            console.Output("learningRate: Learning rate");
                            break;
                        case "accuracy":
                            console.Output("accuracy - Display the accuracy of the model");
                            break;
                        case "save":
                            console.Output("save <filename> - Save the model to a file");
                            console.Output("filename: Name of the file to save the model to");
                            break;
                        case "load":
                            console.Output("load <filename> - Load the model from a file");
                            console.Output("filename: Name of the file to load the model from");
                            break;
                        case "predict":
                            console.Output("predict <numWords> <words> - Predict the next words");
                            console.Output("numWords: Number of words to predict");
                            console.Output("words: Words to predict from");
                            break;
                        case "vocab":
                            console.Output("vocab - Display the vocabulary of the model");
                            break;
                        case "similarity":
                            console.Output("similarity <word1> <word2> - Display the cosine similarity between two words");
                            console.Output("word1: First word");
                            console.Output("word2: Second word");
                            break;
                        case "vector":
                            console.Output("vector <word> - Display the embedding vector of a word");
                            console.Output("word: Word to get the vector of");
                            break;
                        case "findsimilar":
                            console.Output("findsimilar <word> - Display the 5 most similar words to a word");
                            console.Output("word: Word to find similar words to");
                            break;
                        case "add":
                            console.Output("add <word1> <word2> - Add two words and display the closest word to the result");
                            console.Output("word1: First word");
                            console.Output("word2: Second word");
                            break;
                        case "subtract":
                            console.Output("subtract <word1> <word2> - Subtract two words and display the closest word to the result");
                            console.Output("word1: First word");
                            console.Output("word2: Second word");
                            break;
                        case "clear":
                            console.Output("clear - Clear the console");
                            break;
                        case "exit":
                            console.Output("exit - Exit the console");
                            break;
                        default:
                            console.Output("Command not found");
                            break;
                    }
                }
            }
        });
        console.start();
    }
}
