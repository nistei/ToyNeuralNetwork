using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace ToyNeuralNetwork
{
    public delegate double ConverterFunction(double var);
    public class NeuralNetwork
    {
        public int InputNodes { get; private set; }
        public int HiddenLayers { get; private set; }
        public int HiddenNodes { get; private set; }
        public int OutputNodes { get; private set; }

        public Matrix<double>[] Weigths { get; private set; }
        public Matrix<double>[] Biases { get; private set; }

        public double LearningRate { get; set; }
        public ActivationFunction ActivationFunction { get; set; }

        //Constructor
        public NeuralNetwork(int inputNodes = 2, int hiddenLayers = 1, int hiddenNodes = 2, int outputNodes = 1, double learningRate = 0.01, ActivationFunction activationFunction = null)
        {
            InputNodes = inputNodes;
            HiddenLayers = hiddenLayers;
            HiddenNodes = hiddenNodes;
            OutputNodes = outputNodes;

            LearningRate = learningRate;
            if(activationFunction != null)
            {
                ActivationFunction = activationFunction;
            }
            else
            {
                ActivationFunction = ActivationFunction.Sigmoid();
            }

            InitalizeWeights();
            InitializeBiases();
        }

        private void InitalizeWeights()
        {
            Weigths = new Matrix<double>[HiddenLayers + 1];

            for(int i = 0; i < Weigths.Length; i++)
            {
                if(i == 0)
                {
                    Weigths[i] = Matrix<double>.Build.Random(HiddenNodes, InputNodes);
                }
                else if(i == Weigths.Length - 1)
                {
                    Weigths[i] = Matrix<double>.Build.Random(OutputNodes, HiddenNodes);
                }
                else
                {
                    Weigths[i] = Matrix<double>.Build.Random(HiddenNodes, HiddenNodes);
                }
            }
        }

        private void InitializeBiases()
        {
            Biases = new Matrix<double>[HiddenLayers + 1];

            for(int i = 0; i < Biases.Length; i++)
            {
                if(i == Biases.Length - 1)
                {
                    Biases[i] = Matrix<double>.Build.Random(OutputNodes, 1);
                }
                else
                {
                    Biases[i] = Matrix<double>.Build.Random(HiddenNodes, 1);
                }
            }
        }

        private Matrix<double> ArrayToMatrix(double[] input)
        {
            return DenseMatrix.OfColumnMajor(input.Length, 1, input);
        }

        private double[] MatrixToArray(Matrix<double> outputMatrix)
        {
            return outputMatrix.AsRowMajorArray();
        }

        private Matrix<double> CalculateLayer(Matrix<double> weight, Matrix<double> bias, Matrix<double> input)
        {
            Matrix<double> result = weight.Multiply(input);
            result = result.Add(bias);
            return result.Map(c => ActivationFunction.Function(c));
        }

        private Matrix<double> CalculateGradient(Matrix<double> layer, Matrix<double> error)
        {
            Matrix<double> gradient = layer.Map(c => ActivationFunction.DFunction(c));
            gradient = gradient.PointwiseMultiply(error);
            return gradient.Multiply(LearningRate);
        }

        private Matrix<double> CalculateDeltas(Matrix<double> gradient, Matrix<double> layer)
        {
            return gradient.TransposeAndMultiply(layer);
        }

        public double[] Guess(double[] inputArr)
        {
            Matrix<double> output = ArrayToMatrix(inputArr);

            for(int i = 0; i < HiddenLayers + 1; i++)
            {
                output = CalculateLayer(Weigths[i], Biases[i], output);
            }

            return MatrixToArray(output);


            //Matrix<double> hidden = Matrix.op_DotMultiply(WeigthsIH, input);
            //hidden.Add(BiasH);
            //hidden = hidden.Map(c => ActivationFunction.Function(c));
            //
            //Matrix<double> output = Matrix.op_DotMultiply(WeigthsHO, hidden);
            //output.Add(BiasO);
            //output.Map(c => ActivationFunction.Function(c));
            //
            //return output;
        }

        public void Train(double[] inputArr, double[] targetArr)
        {
            Matrix<double> input = ArrayToMatrix(inputArr);
            Matrix<double> target = ArrayToMatrix(targetArr);

            Matrix<double>[] layers = new Matrix<double>[HiddenLayers + 2];
            layers[0] = input;
            for(int i = 1; i < HiddenLayers + 2; i++)
            {
                layers[i] = CalculateLayer(Weigths[i - 1], Biases[i - 1], input);
                input = layers[i];
            }

            for(int i = HiddenLayers + 1; i > 0; i--)
            {
                Matrix<double> error = target.Add(-layers[i]);
                Matrix<double> gradient = CalculateGradient(layers[i], error);
                Matrix<double> delta = CalculateDeltas(gradient, layers[i - 1]);
                Biases[i - 1] = Biases[i - 1].Add(gradient);
                Weigths[i - 1] = Weigths[i - 1].Add(delta);

                Matrix<double> previousError = Weigths[i - 1].Transpose().Multiply(error);
                target = previousError.Add(layers[i - 1]);
            }


            //Matrix<double> hidden = Matrix.op_DotMultiply(WeigthsIH, input);
            //hidden.Add(BiasH);
            //hidden = hidden.Map(c => ActivationFunction.Function(c));
            //
            //Matrix<double> output = Matrix.op_DotMultiply(WeigthsHO, hidden);
            //output.Add(BiasO);
            //output.Map(c => ActivationFunction.Function(c));
            //
            //Matrix<double> outputError = target.Subtract(output);
            //
            //Matrix<double> gradients = output.Map(c => ActivationFunction.DFunction(c));
            //gradients.Multiply(outputError);
            //gradients.Multiply(LearningRate);
            //
            //Matrix<double> hiddenTrans = hidden.Transpose();
            //Matrix<double> weightHODelta = Matrix.op_DotMultiply(gradients, hiddenTrans);
            //
            //WeigthsHO.Add(weightHODelta);
            //BiasO.Add(gradients);
            //
            //Matrix<double> weightHOTrans = hidden.Transpose();
            //Matrix<double> hiddenError = Matrix.op_DotMultiply(weightHOTrans, outputError);
            //
            //Matrix<double> hiddenGradient = hidden.Map(c => ActivationFunction.DFunction(c));
            //hiddenGradient.Multiply(hiddenError);
            //hiddenGradient.Multiply(LearningRate);
            //
            //Matrix<double> inputTrans = input.Transpose();
            //Matrix<double> weightIHDelta = Matrix.op_DotMultiply(hiddenGradient, inputTrans);
            //
            //WeigthsIH.Add(weightIHDelta);
            //BiasH.Add(hiddenGradient);

        }

        //public void Mutate(ConverterFunction func)
        //{
        //    WeigthsIH.Map(c => func(c));
        //    WeigthsHO.Map(c => func(c));
        //    BiasH.Map(c => func(c));
        //    BiasO.Map(c => func(c));
        //}
    }
}
