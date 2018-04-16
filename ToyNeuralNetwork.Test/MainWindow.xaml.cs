using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using ToyNeuralNetwork;

namespace ToyNeuralNetwork.Test
{
    /// <summary>
    /// Interaktionslogik für MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        NeuralNetwork nn = new NeuralNetwork(2, 3, 3, 1, 0.05);

        private List<DataPair> dataSet = new List<DataPair>
        {
            new DataPair(new double[] { 0, 0 }, new double[] { 0 }),
            new DataPair(new double[] { 0, 1 }, new double[] { 1 }),
            new DataPair(new double[] { 1, 0 }, new double[] { 1 }),
            new DataPair(new double[] { 1, 1 }, new double[] { 0 })
        };

        public MainWindow()
        {
            InitializeComponent();

            System.Windows.Threading.DispatcherTimer dispatcherTimer = new System.Windows.Threading.DispatcherTimer();
            dispatcherTimer.Tick += new EventHandler(Draw);
            dispatcherTimer.Interval = new TimeSpan(0, 0, 0, 0, 50);
            dispatcherTimer.Start();
        }

        private void Draw(object sender, EventArgs e)
        {
            Canvas.Children.Clear();

            for (int i = 0; i < 100; i++)
            {
                foreach (DataPair pair in dataSet)
                {
                    nn.Train(pair.Input, pair.Target);
                }
            }

            for (int i = 0; i < Canvas.ActualWidth; i += 10)
            {
                for (int j = 0; j < Canvas.ActualHeight; j += 10)
                {

                    double[] output = nn.Guess(new double[] { i / Canvas.ActualWidth, j / Canvas.ActualHeight });
                    byte colorVal = (byte)((output[0] + 1) * 256);

                    Rectangle rect = new Rectangle();
                    //rect.Stroke = new SolidColorBrush(Colors.Transparent);
                    rect.Width = 10;
                    rect.Height = 10;
                    Canvas.SetLeft(rect, i);
                    Canvas.SetTop(rect, j);

                    rect.Fill = new SolidColorBrush(new Color() { R = colorVal, G = colorVal, B = colorVal, A = 255 });
                    Canvas.Children.Add(rect);
                }
            }
        }
    }
}
