using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ToyNeuralNetwork
{
    public class ActivationFunction
    {
        public ConverterFunction Function { get; set; }
        public ConverterFunction DFunction { get; set; }

        //Constructor
        public ActivationFunction(ConverterFunction func, ConverterFunction dFunc)
        {
            Function = func;
            DFunction = dFunc;
        }

        //Copy-Construtor
        public ActivationFunction(ActivationFunction other)
        {
            Function = other.Function;
            DFunction = other.DFunction;
        }

        public ActivationFunction Clone()
        {
            return new ActivationFunction(this);
        }

        public static ActivationFunction Sigmoid()
        {
            return new ActivationFunction(new ConverterFunction(Sigmoid), new ConverterFunction(DSigmoid));
        }

        public static ActivationFunction TanH()
        {
            return new ActivationFunction(new ConverterFunction(TanH), new ConverterFunction(DTanH));
        }

        private static double TanH(double var)
        {
            return Math.Tanh(var);
        }

        private static double DTanH(double var)
        {
            return 1 - (var * var);
        }

        private static double Sigmoid(double var)
        {
            return 1 / (1 + Math.Exp(-var));
        }

        private static double DSigmoid(double var)
        {
            return var * (1 - var);
        }

    }
}
