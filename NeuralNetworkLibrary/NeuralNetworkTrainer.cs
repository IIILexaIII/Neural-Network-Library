namespace NeuralNetworkLibrary
{
    /// <summary>
    /// Тренер нейронной сети
    /// </summary>
    public class NeuralNetworkTrainer
    {
        private NeuralNetwork Network;

        /// <summary>
        /// Коэффициент обучения
        /// </summary>
        public double learningRatio;
        /// <summary>
        /// Момент
        /// </summary>
        public double moment;

        /// <summary>
        /// Конструктор
        /// </summary>
        /// <param name="Network">Сеть</param>
        /// <param name="learningRatio">Коэффициент обучения</param>
        /// <param name="moment">Момент</param>
        public NeuralNetworkTrainer(NeuralNetwork Network, double learningRatio, double moment)
        {
            this.Network = Network;
            this.learningRatio = learningRatio;
            this.moment = moment;
        }

        /// <summary>
        /// Тренерует сеть
        /// </summary>
        /// <param name="idealValues">Значения, которые должны быть на выходе</param>
        public void Train(double[] idealValues)
        {
            Layer[] layers = Network.layers;
            if (idealValues.Length == layers[layers.Length - 1].neurons.Length)
            {
                // Находим ошибку для выходных нейронов
                for (int neorunIndex = 0; neorunIndex < layers[layers.Length - 1].neurons.Length; neorunIndex++)
                {
                    double outputValue = layers[layers.Length - 1].neurons[neorunIndex].Value;
                    layers[layers.Length - 1].neurons[neorunIndex].Error = idealValues[neorunIndex] - outputValue;
                    // Сразу находим дельту выходных нейронов
                    layers[layers.Length - 1].neurons[neorunIndex].Delta = layers[layers.Length - 1].neurons[neorunIndex].Error * Network.DeriativeFunction(outputValue);
                }
                // Рассчитываем дельту всех нейронов
                for (int layerIndex = layers.Length - 1; layerIndex > 0; layerIndex--)
                    CalculateNeuronDelta(layers[layerIndex - 1], layers[layerIndex]);
                // Корректируем веса
                for (int layerIndex = 0; layerIndex < layers.Length - 1; layerIndex++)
                    WeightAdjustment(layers[layerIndex], layers[layerIndex + 1]);
            }
            else
                throw new System.Exception($"IdealValues.Length ({idealValues.Length}) is not equal to the length of the neurons output layer ({layers[layers.Length - 1].neurons.Length})");
        }

        // Корректировка весов
        private void WeightAdjustment(Layer layerIN, Layer layerOUT)
        {
            for (int neuronIndexOUT = 0; neuronIndexOUT < layerOUT.neurons.Length; neuronIndexOUT++)
            {
                for (int neuronIndexIN = 0; neuronIndexIN < layerIN.neurons.Length; neuronIndexIN++)
                {
                    double grad = layerIN.neurons[neuronIndexIN].Value * layerOUT.neurons[neuronIndexOUT].Delta;
                    double deltaW = learningRatio * grad + moment * layerIN.neurons[neuronIndexIN].DeltaWPrevious[neuronIndexOUT];
                    layerIN.neurons[neuronIndexIN].DeltaWPrevious[neuronIndexOUT] = deltaW;
                    layerIN.neurons[neuronIndexIN].W[neuronIndexOUT] += deltaW;
                }
            }
            if (Network.UseBiasNeurons)
            {
                for (int neuronIndexOUT = 0; neuronIndexOUT < layerOUT.neurons.Length; neuronIndexOUT++)
                {
                    double grad = layerOUT.neurons[neuronIndexOUT].Delta;
                    double deltaW = learningRatio * grad + moment * layerIN.biasNeuron.DeltaWPrevious[neuronIndexOUT];
                    layerIN.biasNeuron.DeltaWPrevious[neuronIndexOUT] = deltaW;
                    layerIN.biasNeuron.W[neuronIndexOUT] += deltaW;
                }
            }
        }

        // Обратное распостранение ошибки
        private void BackpropagationMethod(Layer layerIN, Layer layerOUT)
        {
            for (int neuronIndexIN = 0; neuronIndexIN < layerIN.neurons.Length; neuronIndexIN++)
            {
                layerIN.neurons[neuronIndexIN].Error = 0;
                for (int neuronIndexOUT = 0; neuronIndexOUT < layerOUT.neurons.Length; neuronIndexOUT++)
                {
                    layerIN.neurons[neuronIndexIN].Error += layerIN.neurons[neuronIndexIN].W[neuronIndexOUT] * layerOUT.neurons[neuronIndexOUT].Error;
                }
            }
            // Ошибка для нейрона смещения
            if (Network.UseBiasNeurons)
            {
                layerIN.biasNeuron.Error = 0;
                for (int neuronIndexOUT = 0; neuronIndexOUT < layerOUT.neurons.Length; neuronIndexOUT++)
                {
                    layerIN.biasNeuron.Error += layerIN.biasNeuron.W[neuronIndexOUT] * layerOUT.neurons[neuronIndexOUT].Error;
                }
            }
        }

        // Рассчитываем дельту нейронов
        private void CalculateNeuronDelta(Layer layerIN, Layer layerOUT)
        {
            for (int neuronIndexIN = 0; neuronIndexIN < layerIN.neurons.Length; neuronIndexIN++)
            {
                double sum = 0;
                for (int neuronIndexOUT = 0; neuronIndexOUT < layerOUT.neurons.Length; neuronIndexOUT++)
                    sum += layerIN.neurons[neuronIndexIN].W[neuronIndexOUT] * layerOUT.neurons[neuronIndexOUT].Delta;
                layerIN.neurons[neuronIndexIN].Delta = Network.DeriativeFunction(layerIN.neurons[neuronIndexIN].Value) * sum;
            }
            // Дельта для нейрона смещения
            if (Network.UseBiasNeurons)
            {
                double sum = 0;
                for (int neuronIndexOUT = 0; neuronIndexOUT < layerOUT.neurons.Length; neuronIndexOUT++)
                    sum += layerIN.biasNeuron.W[neuronIndexOUT] * layerOUT.neurons[neuronIndexOUT].Delta;
                layerIN.biasNeuron.Delta = Network.DeriativeFunction(1) * sum;
            }
        }

        /// <summary>
        /// MSE - mean square error - среднеквадратическая ошибка
        /// </summary>
        /// <returns>Среднеквадратическую ошибку выходных нейронов</returns>
        public double GetMSE()
        {
            Layer outputLayer = Network.layers[Network.layers.Length - 1];
            double sumError = 0;
            for (int neuronIndex = 0; neuronIndex < outputLayer.neurons.Length; neuronIndex++)
            {
                Neuron currentNeuron = outputLayer.neurons[neuronIndex];
                sumError += System.Math.Pow(currentNeuron.Error, 2);
            }
            return sumError / outputLayer.neurons.Length;
        }
    }
}