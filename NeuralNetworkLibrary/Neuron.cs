namespace NeuralNetworkLibrary
{
    /// <summary>
    /// Нейрон
    /// </summary>
    public class Neuron
    {
        /// <summary>
        /// Весы
        /// </summary>
        public double[] W;
        /// <summary>
        /// Значение нейрона
        /// </summary>
        public double Value { get; set; }
        /// <summary>
        /// Ошибка нейрона
        /// </summary>
        public double Error { get; set; }
        /// <summary>
        /// Дельта нейрона
        /// </summary>
        public double Delta { get; set; }
        /// <summary>
        /// Предыдущая дельта весов
        /// </summary>
        public double[] DeltaWPrevious;
    }
}