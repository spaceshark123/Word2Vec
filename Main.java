import java.io.*;
import java.util.*;

public class Main {
    public static void main(String[] args) throws Exception {
        clear();
        //load corpus from file corpus.txt (read as one string)
        BufferedReader reader = new BufferedReader(new FileReader("corpus.txt"));
        StringBuilder corpus = new StringBuilder();
        String line;
        while ((line = reader.readLine()) != null) {
            corpus.append(line);
        }
        reader.close();

        //word2vec parameters
        int dimensions = 500;
        int windowSize = 2;
        int minFrequency = 5;

        //create word2vec model
        Word2Vec model = new Word2Vec(Word2Vec.ModelType.CBOW, corpus.toString(), minFrequency, windowSize, dimensions);
        model.getNetwork().displayAccuracy = true;

        //training parameters
        int epochs = 10;
        double learningRate = 0.001;

        //train model
        model.train(epochs, learningRate);
        //print accuracy
        System.out.println("Accuracy: " + model.accuracy() * 100 + "%");
        //save model
        NeuralNetwork.Save(model.getNetwork(), "Word2Vec.model");

        //evaluate model (take user input)
        Scanner scanner = new Scanner(System.in);
        while (true) {
            try {
                //exit or continue
                System.out.println();
                System.out.println("Press enter to continue or type 'exit' to exit");
                String input = scanner.nextLine();
                if (input.equals("exit")) {
                    break;
                }
                System.out.println("Enter words as input (separated by spaces):");
                String[] words = model.cleanText(scanner.nextLine()).split(" ");
                //how many words to predict
                System.out.println("Enter number of words to predict:");
                int numWords = scanner.nextInt();
                scanner.nextLine();
                //print predictions with probabilities beside them
                String[] predictions = model.predict(numWords, words);
                if(predictions[0].equals("Error")) {
                    continue;
                }
                System.out.println("Predictions:");
                for (String prediction : predictions) {
                    int index = model.wordIndex(prediction);
                    double probability = model.getNetwork().GetNeurons()[model.getNetwork().GetNeurons().length - 1][index];
                    probability = Math.round(probability * 10000.0) / 100.0;
                    System.out.println(prediction + " - " + probability + "%");
                }
            } catch (Exception e) {
                System.out.println("Invalid input");
            }
        }
        scanner.close();
    }

    static void clear() {
        //clear console
        System.out.print("\033[H\033[2J");
        System.out.flush();
    }
}
