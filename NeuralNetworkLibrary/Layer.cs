namespace NeuralNetworkLibrary
{
    /// <summary>
    /// Слой
    /// </summary>
    public class Layer
    {
        /// <summary>
        /// Нейрон смещения
        /// </summary>
        public Neuron biasNeuron;
        /// <summary>
        /// Нейроны
        /// </summary>
        public Neuron[] neurons;

        /// <summary>
        /// Конструктор
        /// </summary>
        /// <param name="neuronsCount">Количество нейронов в слое</param>
        public Layer(int neuronsCount)
        {
            biasNeuron = new Neuron();
            neurons = new Neuron[neuronsCount];
            for (int i = 0; i < neuronsCount; i++)
                neurons[i] = new Neuron();
        }
    }
}