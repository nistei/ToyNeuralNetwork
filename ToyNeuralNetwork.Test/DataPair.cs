using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ToyNeuralNetwork.Test
{
    public class DataPair
    {
        public double[] Input { get; set; }
        public double[] Target { get; set; }

        public DataPair(double[] input, double[] target)
        {
            Input = input;
            Target = target;
        }

    }
}
