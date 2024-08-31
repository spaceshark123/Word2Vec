import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.Collections;
import java.io.FileWriter;
import java.io.BufferedReader;
import java.io.FileReader;

import java.util.stream.IntStream;
import java.util.concurrent.atomic.AtomicInteger;

public class NeuralNetwork implements Serializable {
	private static final long serialVersionUID = 1L;

	//first dimension is layer, second dimension is neuron # in layer
	protected double[][] neurons;
	protected double[][] neuronsRaw;
	protected double[][] biases;
	//first dimension is recieving layer, second dimension is recieving neuron #, third dimension is incoming neuron # from previous layer
	protected double[][][] weights;
	protected int[] neuronsPerLayer;
	/* activation choices:
		- linear
		- sigmoid
		- tanh
		- relu
		- binary
		- softmax
	*/
	protected String[] activations;
	public int numLayers;
	//gradient clipping threshold
	public double clipThreshold = 1;
	//whether or not to display accuracy while training (for classification models)
	public boolean displayAccuracy = false;
	public boolean enableBiases = true;

	//extra variables for word2vec data
	public String modelType;
	public int minFrequency;
	public int windowSize;
	public int dimensions;
	public String corpus;
	// interval between training progress bar updates
	public int progressBarInterval = 1;

	// Regularization type
	public static enum RegularizationType {
		NONE,
		L1,
		L2
	}

	// Callback interface for training updates
	public static interface TrainingCallback {
		//test accuracy is assigned -1 if not available for the current mini-batch
		void onEpochUpdate(int epoch, int batch, double progress, double trainAccuracy, double testAccuracy);
	}

	public static interface Optimizer {
		void initialize(int[] neuronsPerLayer, double[][] biases, double[][][] weights);
		void step(double[][] avgBiasGradient, double[][][] avgWeightGradient, double learningRate);
	}

	public static class OptimizerType {
		public static class SGD implements Optimizer {
			private double[][] biases;
			private double[][][] weights;
			private int[] neuronsPerLayer;

			@Override
			public void initialize(int[] neuronsPerLayer, double[][] biases, double[][][] weights) {
				this.biases = biases;
				this.weights = weights;
				this.neuronsPerLayer = neuronsPerLayer;
			}

			@Override
			public void step(double[][] avgBiasGradient, double[][][] avgWeightGradient, double learningRate) {
				for (int i = 1; i < neuronsPerLayer.length; i++) {
					for (int j = 0; j < neuronsPerLayer[i]; j++) {
						//apply velocity
						biases[i][j] = biases[i][j] - learningRate * avgBiasGradient[i][j];
						for (int k = 0; k < neuronsPerLayer[i - 1]; k++) {
							//apply velocity
							weights[i][j][k] = weights[i][j][k] - learningRate * avgWeightGradient[i][j][k];
						}
					}
				}
			}
		}

		public static class SGDMomentum implements Optimizer {
			private double[][] biases;
			private double[][][] weights;
			private int[] neuronsPerLayer;
			private double[][] biasVelocity;
			private double[][][] weightVelocity;
			private double momentum;

			public SGDMomentum(double momentum) {
				this.momentum = momentum;
			}

			@Override
			public void initialize(int[] neuronsPerLayer, double[][] biases, double[][][] weights) {
				this.biases = biases;
				this.weights = weights;
				this.neuronsPerLayer = neuronsPerLayer;
				biasVelocity = new double[biases.length][biases[0].length];
				weightVelocity = new double[weights.length][weights[0].length][weights[0][0].length];
			}

			@Override
			public void step(double[][] avgBiasGradient, double[][][] avgWeightGradient, double learningRate) {
				for (int i = 1; i < neuronsPerLayer.length; i++) {
					for (int j = 0; j < neuronsPerLayer[i]; j++) {
						//do momentum
						biasVelocity[i][j] = momentum * biasVelocity[i][j] + (1 - momentum) * avgBiasGradient[i][j];
						//apply velocity
						biases[i][j] = biases[i][j] - learningRate * biasVelocity[i][j];
						for (int k = 0; k < neuronsPerLayer[i - 1]; k++) {
							//do momentum
							weightVelocity[i][j][k] = momentum * weightVelocity[i][j][k] + (1 - momentum) * avgWeightGradient[i][j][k];
							//apply velocity
							weights[i][j][k] = weights[i][j][k] - learningRate * weightVelocity[i][j][k];
						}
					}
				}
			}
		}

		public static class AdaGrad implements Optimizer {
			private double[][] biases;
			private double[][][] weights;
			private int[] neuronsPerLayer;
			private double[][] biasCache;
			private double[][][] weightCache;
			private double epsilon = 1e-8;

			@Override
			public void initialize(int[] neuronsPerLayer, double[][] biases, double[][][] weights) {
				this.biases = biases;
				this.weights = weights;
				this.neuronsPerLayer = neuronsPerLayer;
				biasCache = new double[biases.length][biases[0].length];
				weightCache = new double[weights.length][weights[0].length][weights[0][0].length];
			}

			@Override
			public void step(double[][] avgBiasGradient, double[][][] avgWeightGradient, double learningRate) {
				for (int i = 1; i < neuronsPerLayer.length; i++) {
					for (int j = 0; j < neuronsPerLayer[i]; j++) {
						//update cache
						biasCache[i][j] += avgBiasGradient[i][j] * avgBiasGradient[i][j];
						//apply update
						biases[i][j] = biases[i][j] - learningRate * avgBiasGradient[i][j] / (Math.sqrt(biasCache[i][j]) + epsilon);
						for (int k = 0; k < neuronsPerLayer[i - 1]; k++) {
							//update cache
							weightCache[i][j][k] += avgWeightGradient[i][j][k] * avgWeightGradient[i][j][k];
							//apply update
							weights[i][j][k] = weights[i][j][k] - learningRate * avgWeightGradient[i][j][k] / (Math.sqrt(weightCache[i][j][k]) + epsilon);
						}
					}
				}
			}
		}

		public static class RMSProp implements Optimizer {
			private double[][] biases;
			private double[][][] weights;
			private int[] neuronsPerLayer;
			private double[][] biasCache;
			private double[][][] weightCache;
			private double decayRate;
			private double epsilon = 1e-8;

			public RMSProp(double decayRate) {
				this.decayRate = decayRate;
			}

			@Override
			public void initialize(int[] neuronsPerLayer, double[][] biases, double[][][] weights) {
				this.biases = biases;
				this.weights = weights;
				this.neuronsPerLayer = neuronsPerLayer;
				biasCache = new double[biases.length][biases[0].length];
				weightCache = new double[weights.length][weights[0].length][weights[0][0].length];
			}

			@Override
			public void step(double[][] avgBiasGradient, double[][][] avgWeightGradient, double learningRate) {
				for (int i = 1; i < neuronsPerLayer.length; i++) {
					for (int j = 0; j < neuronsPerLayer[i]; j++) {
						//update cache
						biasCache[i][j] = decayRate * biasCache[i][j] + (1 - decayRate) * avgBiasGradient[i][j] * avgBiasGradient[i][j];
						//apply update
						biases[i][j] = biases[i][j] - learningRate * avgBiasGradient[i][j] / (Math.sqrt(biasCache[i][j]) + epsilon);
						for (int k = 0; k < neuronsPerLayer[i - 1]; k++) {
							//update cache
							weightCache[i][j][k] = decayRate * weightCache[i][j][k] + (1 - decayRate) * avgWeightGradient[i][j][k] * avgWeightGradient[i][j][k];
							//apply update
							weights[i][j][k] = weights[i][j][k] - learningRate * avgWeightGradient[i][j][k] / (Math.sqrt(weightCache[i][j][k]) + epsilon);
						}
					}
				}
			}
		}

		public static class Adam implements Optimizer {
			private double[][] biases;
			private double[][][] weights;
			private int[] neuronsPerLayer;
			private double[][] biasM;
			private double[][] biasV;
			private double[][][] weightM;
			private double[][][] weightV;
			private double beta1;
			private double beta2;
			private double epsilon = 1e-8;
			private double beta1t = 1;
			private double beta2t = 1;

			public Adam(double beta1, double beta2) {
				this.beta1 = beta1;
				this.beta2 = beta2;
			}

			@Override
			public void initialize(int[] neuronsPerLayer, double[][] biases, double[][][] weights) {
				this.biases = biases;
				this.weights = weights;
				this.neuronsPerLayer = neuronsPerLayer;
				biasM = new double[biases.length][biases[0].length];
				biasV = new double[biases.length][biases[0].length];
				weightM = new double[weights.length][weights[0].length][weights[0][0].length];
				weightV = new double[weights.length][weights[0].length][weights[0][0].length];
			}

			@Override
			public void step(double[][] avgBiasGradient, double[][][] avgWeightGradient, double learningRate) {
				beta1t *= beta1;
				beta2t *= beta2;
				double biasCorrectedM;
				double biasCorrectedV;
				double weightCorrectedM;
				double weightCorrectedV;
				for (int i = 1; i < neuronsPerLayer.length; i++) {
					for (int j = 0; j < neuronsPerLayer[i]; j++) {
						//update biased first moment estimate
						biasM[i][j] = beta1 * biasM[i][j] + (1 - beta1) * avgBiasGradient[i][j];
						//update biased second raw moment estimate
						biasV[i][j] = beta2 * biasV[i][j] + (1 - beta2) * avgBiasGradient[i][j] * avgBiasGradient[i][j];
						//correct bias first moment
						biasCorrectedM = biasM[i][j] / (1 - beta1t);
						//correct bias second moment
						biasCorrectedV = biasV[i][j] / (1 - beta2t);
						//apply update
						biases[i][j] = biases[i][j] - learningRate * biasCorrectedM / (Math.sqrt(biasCorrectedV) + epsilon);
						for (int k = 0; k < neuronsPerLayer[i - 1]; k++) {
							//update biased first moment estimate
							weightM[i][j][k] = beta1 * weightM[i][j][k] + (1 - beta1) * avgWeightGradient[i][j][k];
							//update biased second raw moment estimate
							weightV[i][j][k] = beta2 * weightV[i][j][k] + (1 - beta2) * avgWeightGradient[i][j][k] * avgWeightGradient[i][j][k];
							//correct bias first moment
							weightCorrectedM = weightM[i][j][k] / (1 - beta1t);
							//correct bias second moment
							weightCorrectedV = weightV[i][j][k] / (1 - beta2t);
							//apply update
							weights[i][j][k] = weights[i][j][k] - learningRate * weightCorrectedM / (Math.sqrt(weightCorrectedV) + epsilon);
						}
					}
				}
			}
		}
	}

	// Regularization settings (lambda = regularization strength)
	protected double lambda = 0;
	protected RegularizationType regularizationType = RegularizationType.NONE;
	//used in gradient descent
	volatile protected double[][] avgBiasGradient;
	volatile protected double[][][] avgWeightGradient;
	protected Random r;

	//takes in int[] for number of neurons in each layer and string[] for activations of each layer
	public NeuralNetwork(int[] topology, String[] active, String modelType, int minFrequency, int windowSize, int dimensions, String corpus) {
		int maxLayerSize = max(topology);
		neuronsPerLayer = topology.clone();
		numLayers = topology.length;
		neurons = new double[numLayers][maxLayerSize];
		neuronsRaw = new double[numLayers][maxLayerSize];
		biases = new double[numLayers][maxLayerSize];
		weights = new double[numLayers][maxLayerSize][maxLayerSize];
		activations = active.clone();

		this.modelType = modelType;
		this.minFrequency = minFrequency;
		this.windowSize = windowSize;
		this.dimensions = dimensions;
		this.corpus = corpus;
		r = new Random();
	}

	public NeuralNetwork(int[] topology, String[] active, RegularizationType regularizationType,
			double regularizationStrength, String modelType, int minFrequency, int windowSize, int dimensions, String corpus) {
		this(topology, active, modelType, minFrequency, windowSize, dimensions, corpus);
		//set regularization
		this.regularizationType = regularizationType;
		lambda = regularizationStrength;
	}

	public NeuralNetwork() {

	}

	//initialize network with random starting values
	public void Init(double biasSpread) {
		ClearNeurons();
		InitWeights();
		if(enableBiases)
			InitBiases(biasSpread);
	}

	//initialize network with random starting values using a specified weight initialization method ('he' or 'xavier')
	public void Init(String weightInitMethod, double biasSpread) {
		ClearNeurons();
		InitWeights(weightInitMethod);
		if(enableBiases)
			InitBiases(biasSpread);
	}

	void InitWeights(String initMethod) {
		//initMethod is either "he" or "xavier"
		if (initMethod.equals("he")) {
			Random r = new Random();
			for (int i = 1; i < numLayers; i++) {
				int n = neuronsPerLayer[i - 1];
				//he weight initialization (for relu) (gaussian distribution)
				double mean = 0, std = Math.sqrt(2.0 / n);
				for (int j = 0; j < neuronsPerLayer[i]; j++) {
					for (int k = 0; k < neuronsPerLayer[i - 1]; k++) {
						weights[i][j][k] = r.nextGaussian() * std + mean;
					}
				}

			}
		} else if (initMethod.equals("xavier")) {
			for (int i = 1; i < numLayers; i++) {
				int n = neuronsPerLayer[i - 1];
				double min, max;
				//xavier weight initialization (for linear, sigmoid, tanh, etc.) (uniform distribution)
				max = 1 / Math.sqrt(n);
				min = -max;
				for (int j = 0; j < neuronsPerLayer[i]; j++) {
					for (int k = 0; k < neuronsPerLayer[i - 1]; k++) {
						weights[i][j][k] = randDouble(min, max);
					}
				}
			}
		} else {
			InitWeights();
		}
	}

	void InitWeights() {
		for (int i = 1; i < numLayers; i++) {
			int n = neuronsPerLayer[i - 1];
			double min, max;
			if (activations[i].equals("relu")) {
				//he weight initialization (for relu) (gaussian distribution)
				Random r = new Random();
				double mean = 0, std = Math.sqrt(2.0 / n);
				for (int j = 0; j < neuronsPerLayer[i]; j++) {
					for (int k = 0; k < neuronsPerLayer[i - 1]; k++) {
						weights[i][j][k] = r.nextGaussian() * std + mean;
					}
				}
			} else {
				//xavier weight initialization (for linear, sigmoid, tanh, etc.) (uniform distribution)
				max = 1 / Math.sqrt(n);
				min = -max;
				for (int j = 0; j < neuronsPerLayer[i]; j++) {
					for (int k = 0; k < neuronsPerLayer[i - 1]; k++) {
						weights[i][j][k] = randDouble(min, max);
					}
				}
			}
		}
	}

	void InitBiases(double spread) {
		for (int i = 1; i < numLayers; i++) {
			for (int j = 0; j < neuronsPerLayer[i]; j++) {
				biases[i][j] = randDouble(-spread, spread);
			}
		}
	}

	public double[][][] GetWeights() {
		return weights;
	}

	public void SetWeight(int layer, int outgoing, int incoming, double value) {
		weights[layer][outgoing][incoming] = value;
	}

	public double[][] GetBiases() {
		return biases;
	}

	public void SetBias(int layer, int neuron, double bias) {
		biases[layer][neuron] = bias;
	}

	public String[] GetActivations() {
		return activations;
	}

	public void SetActivation(int layer, String act) {
		activations[layer] = act;
	}

	public double[][] GetNeurons() {
		return neurons;
	}

	public int[] GetTopology() {
		return neuronsPerLayer;
	}

	public void SetRegularizationType(RegularizationType regularizationType) {
		this.regularizationType = regularizationType;
	}

	public void SetRegularizationLambda(double lambda) {
		this.lambda = lambda;
	}

	public RegularizationType GetRegularizationType() {
		return regularizationType;
	}

	public double GetRegularizationLambda() {
		return lambda;
	}

	double randDouble(double min, double max) {
		return min + (max - min) * r.nextDouble();
	}

	int max(int[] arr) {
		int m = -1;
		for (int i : arr) {
			if (i > m) {
				m = i;
			}
		}
		return m;
	}

	// Total regularization term calculation
	double regularizationTerm() {
		double regTerm = 0.0;
		if (regularizationType == RegularizationType.L1) {
			// L1 regularization
			for (int i = 1; i < numLayers; i++) {
				for (int j = 0; j < neuronsPerLayer[i]; j++) {
					for (int k = 0; k < neuronsPerLayer[i - 1]; k++) {
						regTerm += Math.abs(weights[i][j][k]);
					}
				}
			}
		} else if (regularizationType == RegularizationType.L2) {
			// L2 regularization
			for (int i = 1; i < numLayers; i++) {
				for (int j = 0; j < neuronsPerLayer[i]; j++) {
					for (int k = 0; k < neuronsPerLayer[i - 1]; k++) {
						regTerm += weights[i][j][k] * weights[i][j][k];
					}
				}
			}
		}
		regTerm *= lambda;
		return regTerm;
	}

	double linear_activation(double raw) {
		return raw;
	}

	double sigmoid_activation(double raw) {
		return 1d / (1 + Math.exp(-raw));
	}

	double tanh_activation(double raw) {
		return Math.tanh(raw);
	}

	double relu_activation(double raw) {
		return Math.max(0, raw);
	}

	double binary_activation(double raw) {
		return raw > 0 ? 1 : 0;
	}

	// cache is used to store the max value of the layer and the total sum of exponentials to avoid recomputation
	double softmax_activation(double raw, double[] neuronValues, double[] cache) {
		if (cache == null) {
			cache = new double[2];
			// Compute the max value of the layer
			cache[0] = max(neuronValues);
			// Compute the normalization factor (sum of exponentials)
			for (double value : neuronValues) {
				cache[1] += Math.exp(value - cache[0]);
			}
		}
		double maxVal = cache[0];

		// Compute the normalization factor (sum of exponentials)
		double total = cache[1];
		// Compute the softmax activation
		return Math.exp(raw - maxVal - Math.log(total));
	}

	double softmax_der(double[] neuronValues, int index, double[] cache) {
		double softmax = softmax_activation(neuronValues[index], neuronValues, cache);
		return softmax * (1.0 - softmax);
	}

	double activate(double raw, int layer, double[] cache) {
		switch (activations[layer]) {
			case "linear":
				return linear_activation(raw);
			case "sigmoid":
				return sigmoid_activation(raw);
			case "tanh":
				return tanh_activation(raw);
			case "relu":
				return relu_activation(raw);
			case "binary":
				return binary_activation(raw);
			case "softmax":
				return softmax_activation(raw, Arrays.copyOfRange(neuronsRaw[layer], 0, neuronsPerLayer[layer]), cache);
			default:
				return linear_activation(raw);
		}
	}

	double activate(double raw, int layer, double[] neuronsRaw, double[] cache) {
		switch (activations[layer]) {
			case "linear":
				return linear_activation(raw);
			case "sigmoid":
				return sigmoid_activation(raw);
			case "tanh":
				return tanh_activation(raw);
			case "relu":
				return relu_activation(raw);
			case "binary":
				return binary_activation(raw);
			case "softmax":
				return softmax_activation(raw, neuronsRaw, cache);
			default:
				return linear_activation(raw);
		}
	}

	double activate_der(double raw, int layer, int index, double[] cache) {
		double val;
		switch (activations[layer]) {
			case "linear":
				return 1;
			case "sigmoid":
				double sigmoidVal = sigmoid_activation(raw);
				val = sigmoidVal * (1 - sigmoidVal);
				// if (Double.isNaN(val)) {
				// 	System.out.println("NaN error in sigmoid activation der");
				// }
				return val;
			case "tanh":
				val = Math.pow(1d / Math.cosh(raw), 2);
				// if (Double.isNaN(val)) {
				// 	System.out.println("NaN error in tanh activation der");
				// }
				return val;
			case "relu":
				if (raw <= 0) {
					return 0;
				} else {
					return 1;
				}
			case "binary":
				return 0;
			case "softmax":
				val = softmax_der(Arrays.copyOfRange(neuronsRaw[layer], 0, neuronsPerLayer[layer]), index, cache);
				// if (Double.isNaN(val)) {
				// 	System.out.println("NaN error in softmax activation der");
				// }
				return val;
			default:
				return 1;
		}
	}

	double activate_der(double raw, int layer, double[] neuronsRaw, int index, double[] cache) {
		double val;
		switch (activations[layer]) {
			case "linear":
				return 1;
			case "sigmoid":
				double sigmoidVal = sigmoid_activation(raw);
				val = sigmoidVal * (1 - sigmoidVal);
				// if (Double.isNaN(val)) {
				// 	System.out.println("NaN error in sigmoid activation der");
				// }
				return val;
			case "tanh":
				val = Math.pow(1d / Math.cosh(raw), 2);
				// if (Double.isNaN(val)) {
				// 	System.out.println("NaN error in tanh activation der");
				// }
				return val;
			case "relu":
				if (raw <= 0) {
					return 0;
				} else {
					return 1;
				}
			case "binary":
				return 0;
			case "softmax":
				val = softmax_der(neuronsRaw, index, cache);
				// if (Double.isNaN(val)) {
				// 	System.out.println("NaN error in softmax activation der");
				// }
				return val;
			default:
				return 1;
		}
	}

	void ClearNeurons() {
		for (int i = 0; i < numLayers; i++) {
			for (int j = 0; j < neurons[i].length; j++) {
				neurons[i][j] = 0;
				neuronsRaw[i][j] = 0;
			}
		}
	}

	void ClearNeurons(double[][] neurons, double[][] neuronsRaw) {
		for (int i = 0; i < numLayers; i++) {
			for (int j = 0; j < neurons[i].length; j++) {
				neurons[i][j] = 0;
				neuronsRaw[i][j] = 0;
			}
		}
	}

	public double[] Evaluate(double[] input, double[][] neurons, double[][] neuronsRaw) {
		ClearNeurons(neurons, neuronsRaw);

		// Set input neurons
        IntStream.range(0, input.length).parallel().forEach(i -> neurons[0][i] = input[i]);

		// Feed forward
        for (int layer = 1; layer < numLayers; layer++) {
            final int currentLayer = layer;  // Capture the current value of layer
            IntStream.range(0, neuronsPerLayer[currentLayer]).parallel().forEach(neuron -> {
                double raw = enableBiases ? biases[currentLayer][neuron] : 0;
                for (int prevNeuron = 0; prevNeuron < neuronsPerLayer[currentLayer - 1]; prevNeuron++) {
                    raw += weights[currentLayer][neuron][prevNeuron] * neurons[currentLayer - 1][prevNeuron];
                }
                neuronsRaw[currentLayer][neuron] = raw;
                if (activations[currentLayer].equals("softmax")) {
                    neurons[currentLayer][neuron] = raw;
                } else {
                    neurons[currentLayer][neuron] = activate(raw, currentLayer, null);
                }
            });

			if (activations[currentLayer].equals("softmax")) {
				double[] cache = null;
				if(neuronsPerLayer[currentLayer] < 1) {
					continue;
				}
				neurons[currentLayer][0] = activate(neuronsRaw[currentLayer][0], currentLayer,
						Arrays.copyOfRange(neuronsRaw[currentLayer], 0, neuronsPerLayer[currentLayer]), cache);
                if(neuronsPerLayer[currentLayer] < 2) {
					continue;
				}
				IntStream.range(1, neuronsPerLayer[currentLayer]).parallel().forEach(i -> {
                    neurons[currentLayer][i] = activate(neuronsRaw[currentLayer][i], currentLayer, Arrays.copyOfRange(neuronsRaw[currentLayer], 0, neuronsPerLayer[currentLayer]), cache);
				});
            }
        }

		//return output layer
		return Arrays.copyOfRange(neurons[numLayers - 1], 0, neuronsPerLayer[numLayers - 1]);
	}

	public double[] Evaluate(double[] input) {
		ClearNeurons();

		// Set input neurons
        IntStream.range(0, input.length).parallel().forEach(i -> neurons[0][i] = input[i]);

        // Feed forward
        for (int layer = 1; layer < numLayers; layer++) {
            final int currentLayer = layer;  // Capture the current value of layer
            IntStream.range(0, neuronsPerLayer[currentLayer]).parallel().forEach(neuron -> {
                double raw = enableBiases ? biases[currentLayer][neuron] : 0;
                for (int prevNeuron = 0; prevNeuron < neuronsPerLayer[currentLayer - 1]; prevNeuron++) {
                    raw += weights[currentLayer][neuron][prevNeuron] * neurons[currentLayer - 1][prevNeuron];
                }
                neuronsRaw[currentLayer][neuron] = raw;
                if (activations[currentLayer].equals("softmax")) {
                    neurons[currentLayer][neuron] = raw;
                } else {
                    neurons[currentLayer][neuron] = activate(raw, currentLayer, null);
                }
            });

            if (activations[currentLayer].equals("softmax")) {
                double[] cache = null;
				if(neuronsPerLayer[currentLayer] < 1) {
					continue;
				}
				neurons[currentLayer][0] = activate(neuronsRaw[currentLayer][0], currentLayer,
						Arrays.copyOfRange(neuronsRaw[currentLayer], 0, neuronsPerLayer[currentLayer]), cache);
                if(neuronsPerLayer[currentLayer] < 2) {
					continue;
				}
				IntStream.range(1, neuronsPerLayer[currentLayer]).parallel().forEach(i -> {
                    neurons[currentLayer][i] = activate(neuronsRaw[currentLayer][i], currentLayer, Arrays.copyOfRange(neuronsRaw[currentLayer], 0, neuronsPerLayer[currentLayer]), cache);
				});
            }
        }

		//return output layer
		return Arrays.copyOfRange(neurons[numLayers - 1], 0, neuronsPerLayer[numLayers - 1]);
	}

	@Override
	public String toString() {
		StringBuilder print = new StringBuilder().append("Neural Network \n");
		print.append("\nTopology (neurons per layer): ").append(printArr(neuronsPerLayer));
		print.append("\nActivations (per layer): ").append(printArr(activations));
		print.append("\nRegularization: ").append(regularizationType.toString()).append(" lambda: ").append(lambda);

		print.append("\nBiases:\n");
		for (int i = 0; i < numLayers; i++) {
			print.append("Layer ").append((i + 1)).append(": ").append(printArr(Arrays.copyOfRange(biases[i], 0, neuronsPerLayer[i]))).append("\n");
		}

		print.append("\nWeights:\n");
		for (int i = 1; i < numLayers; i++) {
			for (int j = 0; j < neuronsPerLayer[i]; j++) {
				//each neuron
				print.append("    Neuron ").append((j + 1)).append(" of Layer ").append((i + 1)).append(" Weights: \n").append(printArr(Arrays.copyOfRange(weights[i][j], 0, neuronsPerLayer[i - 1]))).append("\n");
			}
		}
		return print.toString();
	}

	String printArr(int[] arr) {
		if (arr == null)
			return "[]";
		if (arr.length == 0)
			return "[]";
		StringBuilder print = new StringBuilder().append("[");
		for (int i = 0; i < arr.length - 1; i++) {
			print.append(arr[i]).append(", ");
		}
		print.append(arr[arr.length - 1]).append("]");
		return print.toString();
	}

	String printArr(double[] arr) {
		if (arr == null)
			return "[]";
		if (arr.length == 0)
			return "[]";
		StringBuilder print = new StringBuilder().append("[");
		for (int i = 0; i < arr.length - 1; i++) {
			print.append(arr[i]).append(", ");
		}
		print.append(arr[arr.length - 1]).append("]");
		return print.toString();
	}

	String printArr(String[] arr) {
		if (arr == null)
			return "[]";
		if (arr.length == 0)
			return "[]";
		StringBuilder print = new StringBuilder().append("[");
		for (int i = 0; i < arr.length - 1; i++) {
			print.append(arr[i]).append(", ");
		}
		print.append(arr[arr.length - 1]).append("]");
		return print.toString();
	}

	// save the neural network to a file directly as a java object
	// not transferable between different programming languages and not human readable
	public static void Save(NeuralNetwork network, String path) {
		try {
			FileOutputStream f = new FileOutputStream(path);
			ObjectOutputStream o = new ObjectOutputStream(f);

			// Write objects to file
			o.writeObject(network);

			o.close();
			f.close();
		} catch (FileNotFoundException e) {
			System.out.println("File not found");
		} catch (IOException e) {
			System.out.println("Error initializing stream");
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	// save the neural network to a file as a plain text file
	// transferable between different programming languages and human readable
	public static void SaveParameters(NeuralNetwork network, String path) {
		BufferedWriter writer = null;
		try {
			FileWriter fWriter = new FileWriter(path);
			writer = new BufferedWriter(fWriter);
			StringBuilder print = new StringBuilder();
			//write parameters to print
			print.append("numlayers ").append(network.numLayers).append("\n");
			print.append("topology ");
			for (int i = 0; i < network.neuronsPerLayer.length; i++) {
				print.append(network.neuronsPerLayer[i]).append(" ");
			}
			print.append("\nactivations ");
			for (int i = 0; i < network.activations.length; i++) {
				print.append(network.activations[i]).append(" ");
			}
			print.append("\nregularization ").append(network.regularizationType.toString()).append(" ")
					.append(network.lambda).append("\n");
			print.append("biases ");
			for (int i = 0; i < network.biases.length; i++) {
				for (int j = 0; j < network.neuronsPerLayer[i]; j++) {
					print.append(network.biases[i][j]).append(" ");
				}
			}
			//weights start at layer 1 because layer 0 is the input layer
			print.append("\nweights ");
			for (int i = 1; i < network.weights.length; i++) {
				for (int j = 0; j < network.neuronsPerLayer[i]; j++) {
					for (int k = 0; k < network.neuronsPerLayer[i - 1]; k++) {
						print.append(network.weights[i][j][k]).append(" ");
					}
				}
			}
			//write to file
			writer.write(print.toString());
			writer.close();
		} catch (FileNotFoundException e) {
			System.out.println("File not found");
		} catch (IOException e) {
			System.out.println("Error initializing stream");
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	// load a neural network from a file that was saved directly as a java object
	// not transferable between different programming languages
	public static NeuralNetwork Load(String path) {
		try {
			FileInputStream fi = new FileInputStream(path);
			ObjectInputStream oi = new ObjectInputStream(fi);

			// Read objects
			NeuralNetwork loadedNetwork = (NeuralNetwork) oi.readObject();
			loadedNetwork.r = new Random();

			oi.close();
			fi.close();

			return loadedNetwork;
		} catch (FileNotFoundException e) {
			System.out.println("File not found");
		} catch (IOException e) {
			System.out.println("Error initializing stream");
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return null;
	}

	// load a neural network from a file that was saved as a plain text file
	// transferable between different programming languages
	public static NeuralNetwork LoadParameters(String path) {
		try (BufferedReader reader = new BufferedReader(new FileReader(path))) {
			String line;
			NeuralNetwork network = new NeuralNetwork();
			while ((line = reader.readLine()) != null) {
				String[] tokens = line.split(" ");
				String paramType = tokens[0];

				switch (paramType) {
					case "numlayers":
						network.numLayers = Integer.parseInt(tokens[1]);
						network.neuronsPerLayer = new int[network.numLayers];
						network.activations = new String[network.numLayers];
						break;
					case "topology":
						int maxLayerSize = 0;
						for (int i = 1; i < tokens.length; i++) {
							network.neuronsPerLayer[i - 1] = Integer.parseInt(tokens[i]);
							maxLayerSize = Math.max(maxLayerSize, network.neuronsPerLayer[i - 1]);
						}
						network.neurons = new double[network.numLayers][maxLayerSize];
						network.neuronsRaw = new double[network.numLayers][maxLayerSize];
						network.biases = new double[network.numLayers][maxLayerSize];
						network.weights = new double[network.numLayers][maxLayerSize][maxLayerSize];
						break;
					case "activations":
						for (int i = 1; i < tokens.length; i++) {
							network.activations[i - 1] = tokens[i];
						}
						break;
					case "regularization":
						network.regularizationType = RegularizationType.valueOf(tokens[1]);
						network.lambda = Double.parseDouble(tokens[2]);
						break;
					case "biases":
						int layerIndex = 0;
						int neuronIndex = 0;
						for (int i = 1; i < tokens.length; i++) {
							network.biases[layerIndex][neuronIndex] = Double.parseDouble(tokens[i]);
							neuronIndex++;
							if (neuronIndex == network.neuronsPerLayer[layerIndex]) {
								neuronIndex = 0;
								layerIndex++;
							}
						}
						break;
					case "weights":
						layerIndex = 1;
						neuronIndex = 0;
						int incomingNeuronIndex = 0;
						for (int i = 1; i < tokens.length; i++) {
							network.weights[layerIndex][neuronIndex][incomingNeuronIndex] = Double
									.parseDouble(tokens[i]);
							incomingNeuronIndex++;
							if (incomingNeuronIndex == network.neuronsPerLayer[layerIndex - 1]) {
								incomingNeuronIndex = 0;
								neuronIndex++;
								if (neuronIndex == network.neuronsPerLayer[layerIndex]) {
									neuronIndex = 0;
									layerIndex++;
								}
							}
						}
						break;
				}
			}
			network.r = new Random();
			return network;
		} catch (FileNotFoundException e) {
			System.out.println("File not found");
		} catch (IOException e) {
			System.out.println("Error reading from file");
		} catch (ArrayIndexOutOfBoundsException e) {
			System.out.println("File not formatted correctly");
		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
	}

	//chance is a number between 0 and 1
	public void Mutate(double chance, double variation) {
		//mutate weights
		for (int i = 0; i < weights.length; i++) {
			for (int j = 0; j < weights[0].length; j++) {
				for (int k = 0; k < weights[0][0].length; k++) {
					if (randDouble(0, 1) <= chance) {
						weights[i][j][k] += randDouble(-variation, variation);
					}
				}
			}
		}
		//mutate biases
		for (int i = 0; i < biases.length; i++) {
			for (int j = 0; j < biases[0].length; j++) {
				if (randDouble(0, 1) <= chance) {
					biases[i][j] += randDouble(-variation, variation);
				}
			}
		}
	}

	public NeuralNetwork clone() {
		NeuralNetwork clone = new NeuralNetwork(neuronsPerLayer, activations, regularizationType, lambda, modelType,
				minFrequency, windowSize, dimensions, corpus);
		clone.biases = biases.clone();
		clone.weights = weights.clone();
		return clone;
	}

	//error functions
	public double Cost(double[] output, double[] expected, String lossFunction) {
		double cost = 0;
		if (output.length != expected.length) {
			return -1;
		}
		if (lossFunction.equals("sse")) {
			for (int i = 0; i < output.length; i++) {
				double neuronCost = 0.5 * Math.pow(expected[i] - output[i], 2);
				cost += neuronCost;
			}
		} else if (lossFunction.equals("mse")) {
			for (int i = 0; i < output.length; i++) {
				double neuronCost = Math.pow(expected[i] - output[i], 2);
				cost += neuronCost;
			}
			cost /= output.length;
		} else if (lossFunction.equals("categorical_crossentropy")) {
			for (int i = 0; i < output.length; i++) {
				cost -= expected[i] * Math.log(output[i] + 1.0e-15d);
			}
		}
		//add regularization term
		cost += regularizationTerm();
		return cost;
	}

	//error functions derivative
	double cost_der(double predicted, double expected, String lossFunction) {
		if (lossFunction.equals("sse")) {
			// if (Double.isNaN(predicted - expected)) {
			// 	System.out.println("NaN error in cost derivative sse");
			// }
			return predicted - expected;
		} else if (lossFunction.equals("mse")) {
			// if (Double.isNaN((2.0 * (predicted - expected)) / neuronsPerLayer[numLayers - 1])) {
			// 	System.out.println("NaN error in cost derivative mse");
			// }
			return (2.0 * (predicted - expected)) / neuronsPerLayer[numLayers - 1];
		} else if (lossFunction.equals("categorical_crossentropy")) {
			// if (Double.isNaN(-expected / (Math.max(predicted, 1.0e-15)))) {
			// 	System.out.println("NaN error in cost derivative crossentropy: expected: " + expected + " predicted: "
			// 			+ predicted);
			// }
			return -expected / (predicted + 1.0e-15);
		}
		return 1;
	}

	double[] Backpropagate(double[][] biasGrad, double[][][] weightGrad, double[] predicted, double[] expected, String lossFunction) {
		return Backpropagate(this.neurons, this.neuronsRaw, biasGrad, weightGrad, predicted, expected, 1, lossFunction);
	}

	double[] Backpropagate(double[][] neurons, double[][] neuronsRaw, double[][] biasGrad, double[][][] weightGrad, double[] predicted, double[] expected, int layer, String lossFunction) {
		double[] neuronGradients = new double[neuronsPerLayer[layer]];

		//base case
		if (layer == numLayers - 1) {
			//last layer (output layer)
			double[] cache = null;
			if (lossFunction.equals("categorical_crossentropy") && activations[layer].equals("softmax")) {
				for (int i = 0; i < neuronsPerLayer[layer]; i++) {
					// Softmax with categorical crossentropy simplification to speed up computation
					neuronGradients[i] = predicted[i] - expected[i];
					for (int j = 0; j < neuronsPerLayer[layer - 1]; j++) {
						weightGrad[layer][i][j] = neuronGradients[i] * neurons[layer - 1][j];
					}
				}
			} else if(activations[layer].equals("softmax")) {
				for (int i = 0; i < neuronsPerLayer[layer]; i++) {
					neuronGradients[i] = cost_der(predicted[i], expected[i], lossFunction)
								* activate_der(neuronsRaw[layer][i], layer, neuronsRaw[layer], i, cache);
					for (int j = 0; j < neuronsPerLayer[layer - 1]; j++) {
						weightGrad[layer][i][j] = neuronGradients[i] * neurons[layer - 1][j];
					}
				}
			} else {
				for (int i = 0; i < neuronsPerLayer[layer]; i++) {
					neuronGradients[i] = cost_der(predicted[i], expected[i], lossFunction)
									* activate_der(neuronsRaw[layer][i], layer, neuronsRaw[layer], i, null);
					for (int j = 0; j < neuronsPerLayer[layer - 1]; j++) {
						weightGrad[layer][i][j] = neuronGradients[i] * neurons[layer - 1][j];
					}
				}
			}
			return neuronGradients;
		}

		//recursive case
		double[] nextLayerBackpropagate = Backpropagate(neurons, neuronsRaw, biasGrad, weightGrad, predicted, expected, layer + 1, lossFunction);
		//double nextLayerSum = 0;
		double[] nextLayerWeightedSum = new double[neuronsPerLayer[layer]];
		// for (int i = 0; i < neuronsPerLayer[layer + 1]; i++) {
		// 	nextLayerSum += nextLayerBackpropagate[i];
		// }
		for (int i = 0; i < neuronsPerLayer[layer]; i++) {
			for (int j = 0; j < neuronsPerLayer[layer + 1]; j++) {
				nextLayerWeightedSum[i] += nextLayerBackpropagate[j] * weights[layer + 1][j][i];
			}
			neuronGradients[i] = activate_der(neuronsRaw[layer][i], layer, i, null) * nextLayerWeightedSum[i];
			for (int j = 0; j < neuronsPerLayer[layer - 1]; j++) {
				weightGrad[layer][i][j] = neuronGradients[i] * neurons[layer - 1][j];
			}
		}

		//return gradients of neurons in layer
		return neuronGradients;
	}

	public static double clamp(double value, double min, double max) {
		return Math.max(min, Math.min(max, value));
	}

	public double evaluateAccuracy(double[][] inputs, double[][] outputs) {
        int numCorrect = 0;
        for (int i = 0; i < inputs.length; i++) {
            double[] predicted = Evaluate(inputs[i]); // Predict the output for each input
            int prediction = indexOf(predicted, max(predicted));
            int actual = indexOf(outputs[i], max(outputs[i]));
            if (prediction == actual) {
                numCorrect++;
            }
        }
        return (double) numCorrect / inputs.length;
    }

	public void Train(double[][] trainX, double[][] trainY, double[][] testX, double[][] testY, int epochs, double learningRate, int batchSize,
			String lossFunction, double decay, Optimizer optimizer, TrainingCallback callback) {
		double lr = learningRate;
		//list of indices for data points, will be randomized in each epoch
		if (batchSize == -1) {
			batchSize = trainX.length;
		}
		List<Integer> indices = new ArrayList<Integer>(trainX.length);
		for (int i = 0; i < trainX.length; i++) {
			indices.add(i);
		}
		//initial shuffle
		Collections.shuffle(indices);
		//current index of data point
		int currentInd = 0;
		//precompute weighted average (multiply each element by this to average out all data points in batch)
		final double weightedAvg = 1.0 / (double) batchSize;
		int epoch = 0;
		double progress = 0; // marks the epoch and progress through current epoch as decimal
		final int batchesPerEpoch = (int) Math.ceil((double) trainX.length / batchSize);
		int epochIteration = 0;
		if (trainX.length % batchSize != 0) {
			//batches wont divide evenly into samples
			System.out.println("warning: training data size is not divisible by sample size");
		}
		optimizer.initialize(neuronsPerLayer, biases, weights);
		avgBiasGradient = new double[numLayers][biases[0].length];
		avgWeightGradient = new double[numLayers][weights[0].length][weights[0][0].length];
		double avgBatchTime = 0;
		int iteration = 0;
		//ThreadLocal variables for thread-specific arrays
		final ThreadLocal<double[][]> threadLocalNeurons = ThreadLocal.withInitial(() -> new double[numLayers][neurons[0].length]);
		final ThreadLocal<double[][]> threadLocalNeuronsRaw = ThreadLocal.withInitial(() -> new double[numLayers][neurons[0].length]);
		final ThreadLocal<double[][]> threadLocalBiasGradient = ThreadLocal.withInitial(() -> new double[numLayers][neurons[0].length]);
		final ThreadLocal<double[][][]> threadLocalWeightGradient = ThreadLocal.withInitial(() -> {
			double[][][] gradients = new double[numLayers][][];
			for (int i = 0; i < numLayers; i++) {
				gradients[i] = new double[weights[i].length][weights[i][0].length];
			}
			return gradients;
		});
		//initialize accuracies
		double initialTrainAccuracy = 100 * 0;
		double initialTestAccuracy = 100 * 0;
		//round to 2 decimal
		initialTrainAccuracy = Math.round(initialTrainAccuracy * 100.0) / 100.0;
		initialTestAccuracy = Math.round(initialTestAccuracy * 100.0) / 100.0;
		if (callback != null) {
			callback.onEpochUpdate(0, 0, 0, initialTrainAccuracy, initialTestAccuracy);
		}
		AtomicInteger numCorrect = new AtomicInteger(0);
		// Initialize batchIndices once
		ArrayList<Integer> batchIndices = new ArrayList<>(batchSize);

		for (int i = 0; i < batchSize; i++) {
			batchIndices.add(0); // pre-fill with dummy values to avoid resizing
		}
		long startTime = System.currentTimeMillis();
		for (; epoch < epochs; iteration++) {
			//do epoch batch stuff (iteration is the current cumulative batch iteration)
			epochIteration = iteration % batchesPerEpoch;
			//do exponential learning rate decay
			lr = learningRate;

			batchIndices.clear();
			// Use System.arraycopy for faster copying
			int endIndex = currentInd + batchSize;
			if (endIndex <= trainX.length) {
				// If the batch does not wrap around the end of the list
				batchIndices.addAll(indices.subList(currentInd, endIndex));
			} else {
				// If the batch wraps around the end of the list
				int wrapAroundIndex = endIndex % indices.size();
				batchIndices.addAll(indices.subList(currentInd, trainX.length));
				batchIndices.addAll(indices.subList(0, wrapAroundIndex));
			}

			numCorrect.set(0);
			if (iteration > 0) {
				//not first iteration, reset gradients
				for(int i = 1; i < numLayers; i++) {
					for(int j = 0; j < neuronsPerLayer[i]; j++) {
						avgBiasGradient[i][j] = 0;
						for (int k = 0; k < neuronsPerLayer[i-1]; k++) {
							avgWeightGradient[i][j][k] = 0;
						}
					}
				}
			}

			// Parallelize this loop
			IntStream.range(0, batchSize).parallel().forEach(a -> {
				int caseInd = batchIndices.get(a);

				// Use thread-local arrays
				double[][] thisNeurons = threadLocalNeurons.get();
				double[][] thisNeuronsRaw = threadLocalNeuronsRaw.get();
				double[][] thisBiasGradient = threadLocalBiasGradient.get();
				double[][][] thisWeightGradient = threadLocalWeightGradient.get();

				// Calculate predicted output
				double[] predicted = Evaluate(trainX[caseInd], thisNeurons, thisNeuronsRaw);

				// If this is a classification network, count the number correct
				if (displayAccuracy) {
					int prediction = indexOf(predicted, max(predicted));
					int actual = indexOf(trainY[caseInd], max(trainY[caseInd]));
					if (prediction == actual) {
						numCorrect.incrementAndGet();
					}
				}

				// Reset gradients
				for(int i = 1; i < numLayers; i++) {
					for(int j = 0; j < neuronsPerLayer[i]; j++) {
						thisBiasGradient[i][j] = 0;
						for (int k = 0; k < neuronsPerLayer[i-1]; k++) {
							thisWeightGradient[i][j][k] = 0;
						}
					}
				}

				// Do backpropagation
				Backpropagate(thisNeurons, thisNeuronsRaw, thisBiasGradient, thisWeightGradient, predicted,
						trainY[caseInd], 1, lossFunction);

				// Do weighted sum of gradients for average
				for (int i = 1; i < numLayers; i++) {
					for (int j = 0; j < neuronsPerLayer[i]; j++) {
						for (int k = 0; k < neuronsPerLayer[i - 1]; k++) {
							avgWeightGradient[i][j][k] += thisWeightGradient[i][j][k] * weightedAvg;
						}
					}
				}
			});

			// gradient post-processing
			for (int i = 1; i < numLayers; i++) {
				for (int j = 0; j < neuronsPerLayer[i]; j++) {
					// gradient clipping
					avgBiasGradient[i][j] = clamp(avgBiasGradient[i][j], -clipThreshold, clipThreshold);
					for (int k = 0; k < neuronsPerLayer[i - 1]; k++) {
						// regularization
						if (regularizationType == RegularizationType.L1) {
							avgWeightGradient[i][j][k] += lambda * Math.signum(weights[i][j][k]);
						} else if (regularizationType == RegularizationType.L2) {
							avgWeightGradient[i][j][k] += lambda * weights[i][j][k];
						}
						// gradient clipping
						avgWeightGradient[i][j][k] = clamp(avgWeightGradient[i][j][k], -clipThreshold, clipThreshold);
					}
				}
			}
			optimizer.step(avgBiasGradient, avgWeightGradient, lr);
			currentInd += batchSize;
			boolean newEpoch = false;
			if (currentInd >= trainX.length) {
				//new epoch
				newEpoch = true;
				currentInd = 0;
				epoch++;
				Collections.shuffle(indices);
			}
			progress = epoch + currentInd / (double) trainX.length;
			if (displayAccuracy) {
				//calculate train accuracy
				double trainAccuracy = 100 * ((double) numCorrect.get() * weightedAvg);
				//round to 2 decimal
				trainAccuracy = Math.round(trainAccuracy * 100.0) / 100.0;
				//calculate test accuracy if new epoch
				double testAccuracy = -1;
				if (newEpoch) {
					testAccuracy = 100 * 0;
					//round to 2 decimal
					testAccuracy = Math.round(testAccuracy * 100.0) / 100.0;
				}
				if (callback != null) {
					callback.onEpochUpdate(epoch + 1, epochIteration + 1, progress, trainAccuracy, testAccuracy);
				}
				if (progressBarInterval > 0 && iteration % progressBarInterval == 0) {
					progressBar(30, "Training", epoch + 1, epochs,
							(epochIteration + 1) + "/" + batchesPerEpoch + " accuracy: " + trainAccuracy + "%");
				}
			} else {
				if(progressBarInterval > 0 && iteration % progressBarInterval == 0) {
					progressBar(30, "Training", epoch + 1, epochs, (epochIteration + 1) + "/" + batchesPerEpoch);
				}
			}
		}
		long endTime = System.currentTimeMillis();
		avgBatchTime = (endTime - startTime) / (1000.0 * (iteration + 1));
		System.out.println();
		System.out.println("average batch time: " + avgBatchTime + " seconds");
	}

	public void Train(double[][] trainX, double[][] trainY, double[][] testX, double[][] testY, int epochs, double learningRate, int batchSize,
			String lossFunction, double decay, Optimizer optimizer) {
		Train(trainX, trainY, testX, testY, epochs, learningRate, batchSize, lossFunction, decay, optimizer, null);
	}

	public void Train(double[][] trainX, double[][] trainY, double[][] testX, double[][] testY, int epochs, double learningRate, int batchSize,
			String lossFunction, double decay) {
		Train(trainX, trainY, testX, testY, epochs, learningRate, batchSize, lossFunction, decay, new OptimizerType.SGD());
	}

	public void Train(double[][] trainX, double[][] trainY, double[][] testX, double[][] testY, int epochs, double learningRate, int batchSize,
			String lossFunction) {
		Train(trainX, trainY, testX, testY, epochs, learningRate, batchSize, lossFunction, 0);
	}

	public void Train(double[][] trainX, double[][] trainY, double[][] testX, double[][] testY, int epochs, double learningRate, int batchSize,
			String lossFunction, Optimizer optimizer) {
		Train(trainX, trainY, testX, testY, epochs, learningRate, batchSize, lossFunction, 0, optimizer);
	}

	public void progressBar(int width, String title, int current, int total, String subtitle) {
		String filled = "█";
		String unfilled = "░";
		double fill = (double) current / total;
		if (fill >= 0 && fill <= 1) {
			//set progress bar
			int fillAmount = (int) Math.ceil(fill * width);
			StringBuilder bar = new StringBuilder();
			bar.append(title).append(": ").append(filled.repeat(fillAmount)).append(unfilled.repeat(width - fillAmount)).append(" ").append(current).append("/").append(total).append(" ").append(subtitle).append("      ").append("\r");
			System.out.print(bar.toString());
		}
	}

	double max(double[] arr) {
		double m = -1;
		for (double i : arr) {
			if (i > m) {
				m = i;
			}
		}
		return m;
	}

	int indexOf(double[] arr, double v) {
		int index = -1;
		for (int i = 0; i < arr.length; i++) {
			if (arr[i] == v) {
				index = i;
				return index;
			}
		}
		return index;
	}
}