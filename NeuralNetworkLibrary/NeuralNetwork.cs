using System;
using System.IO;
using Newtonsoft.Json;

namespace NeuralNetworkLibrary
{
    /// <summary>
    /// Нейронная сеть
    /// </summary>
    public class NeuralNetwork
    {
        /// <summary>
        /// Делегат функции активации
        /// </summary>
        public delegate double ActivationFunction(double x);
        /// <summary>
        /// Функция активации
        /// </summary>
        public ActivationFunction Function { get; private set; }
        /// <summary>
        /// Производная функции активации
        /// </summary>
        public ActivationFunction DeriativeFunction { get; private set; }

        /// <summary>
        /// Использовать ли нейроны смещения
        /// </summary>
        public bool UseBiasNeurons { get; private set; }

        /// <summary>
        /// Слои нейронной сети
        /// </summary>
        public Layer[] layers;

        /// <summary>
        /// Конструктор
        /// </summary>
        /// <param name="Function">Функция активации</param>
        /// <param name="DeriativeFunction">Производная функции активации</param>
        /// <param name="UseBiasNeurons">Использовать нейроны смещения</param>
        /// <param name="neuronsCount">Количество нейронов в каждом слое</param>
        public NeuralNetwork(ActivationFunction Function, ActivationFunction DeriativeFunction, bool UseBiasNeurons, params int[] neuronsCount)
        {
            this.Function = Function;
            this.DeriativeFunction = DeriativeFunction;
            this.UseBiasNeurons = UseBiasNeurons;
            layers = new Layer[neuronsCount.Length];
            for (int i = 0; i < neuronsCount.Length; i++)
                layers[i] = new Layer(neuronsCount[i]);
            InitializationWeights();
        }

        /// <summary>
        /// Конструктор с загрузкой датабазы
        /// </summary>
        /// <param name="Function">Функция активации</param>
        /// <param name="DeriativeFunction">Производная функции активации</param>
        /// <param name="dataPath">Путь к сохраненному датапаку</param>
        public NeuralNetwork(ActivationFunction Function, ActivationFunction DeriativeFunction, string dataPath)
        {
            this.Function = Function;
            this.DeriativeFunction = DeriativeFunction;
            LoadNetwork(dataPath);
        }

        /// <summary>
        /// Нулевой инициализатор весов (вызывается автоматически при создании экземпляра класса)
        /// </summary>
        public void InitializationWeights()
        {
            for (int layerIndex = 0; layerIndex < layers.Length - 1; layerIndex++)
            {
                // Текущий слой
                Layer currentLayer = layers[layerIndex];
                // Количество нейронов в следующим слое
                int neuronsCountNextLayer = layers[layerIndex + 1].neurons.Length;
                for (int neuronIndex = 0; neuronIndex < currentLayer.neurons.Length; neuronIndex++)
                {
                    currentLayer.neurons[neuronIndex].W = new double[neuronsCountNextLayer];
                    currentLayer.neurons[neuronIndex].DeltaWPrevious = new double[neuronsCountNextLayer];
                }
                // Инициализация нейрона смещения
                if (UseBiasNeurons && layerIndex != layers.Length - 1)
                {
                    currentLayer.biasNeuron.W = new double[neuronsCountNextLayer];
                    currentLayer.biasNeuron.DeltaWPrevious = new double[neuronsCountNextLayer];
                }
            }
        }

        /// <summary>
        /// Случайный инициализатор весов
        /// </summary>
        /// <param name="seed">Зерно</param>
        /// <param name="border">Граница значений весов при инициализации</param>
        public void InitializationWeights(int seed, double border)
        {
            Random rnd = new Random(seed);
            for (int layerIndex = 0; layerIndex < layers.Length - 1; layerIndex++)
            {
                // Текущий слой
                Layer currentLayer = layers[layerIndex];
                // Количество нейронов в следующим слое
                int neuronsCountNextLayer = layers[layerIndex + 1].neurons.Length;
                for (int neuronIndex = 0; neuronIndex < currentLayer.neurons.Length; neuronIndex++)
                {
                    currentLayer.neurons[neuronIndex].W = new double[neuronsCountNextLayer];
                    currentLayer.neurons[neuronIndex].DeltaWPrevious = new double[neuronsCountNextLayer];
                    for (int i = 0; i < currentLayer.neurons[neuronIndex].W.Length; i++)
                        currentLayer.neurons[neuronIndex].W[i] = rnd.NextDouble() * 2 * border - border;
                }
                // Инициализация нейрона смещения
                if (UseBiasNeurons && layerIndex != layers.Length - 1)
                {
                    currentLayer.biasNeuron.W = new double[neuronsCountNextLayer];
                    currentLayer.biasNeuron.DeltaWPrevious = new double[neuronsCountNextLayer];
                }
            }
        }

        // Рассчитать два слоя
        private void CalculateTwoLayers(Layer layerIN, Layer layerOUT)
        {
            for (int neuronIndexOUT = 0; neuronIndexOUT < layerOUT.neurons.Length; neuronIndexOUT++)
            {
                // Сумма весов для нейрона
                double sumW = 0;
                if (UseBiasNeurons)
                    sumW += layerIN.biasNeuron.W[neuronIndexOUT];
                for (int neuronIndexIN = 0; neuronIndexIN < layerIN.neurons.Length; neuronIndexIN++)
                    sumW += layerIN.neurons[neuronIndexIN].W[neuronIndexOUT] * layerIN.neurons[neuronIndexIN].Value;
                layerOUT.neurons[neuronIndexOUT].Value = Function(sumW);
            }
        }

        /// <summary>
        /// Рассчитать все слои
        /// </summary>
        /// <param name="inputValues">Входные значения</param>
        /// <returns>Значение выходных нейронов</returns>
        public double[] CalculateLayers(params double[] inputValues)
        {
            if (inputValues.Length == layers[0].neurons.Length)
            {
                for (int i = 0; i < layers[0].neurons.Length; i++)
                    layers[0].neurons[i].Value = inputValues[i];
                for (int i = 0; i < layers.Length - 1; i++)
                    CalculateTwoLayers(layers[i], layers[i + 1]);

                double[] outputs = new double[layers[layers.Length - 1].neurons.Length];
                for (int i = 0; i < layers[layers.Length - 1].neurons.Length; i++)
                    outputs[i] = layers[layers.Length - 1].neurons[i].Value;
                return outputs;
            }
            else
            {
                throw new Exception($"InputValues.Length ({inputValues.Length}) is not equal to the length of the neurons input layer ({layers[0].neurons.Length})");
            }
        }

        /// <summary>
        /// Функция активации
        /// </summary>
        /// <param name="x">Параметр функции</param>
        /// <returns>Значение от 0 до 1</returns>
        public static double Sigmoid(double x)
        {
            return 1 / (1 + Math.Pow(Math.E, -x));
        }

        /// <summary>
        /// Производная функции активации
        /// </summary>
        /// <param name="x">Значение сигмоиды в точке x</param>
        /// <returns>Значение производной сигмоида</returns>
        public static double DeriativeSigmoid(double x)
        {
            return (1 - x) * x;
        }

        /// <summary>
        /// Функция активации
        /// </summary>
        /// <param name="x">Параметр функции</param>
        /// <returns>Значение от -1 до 1</returns>
        public static double HyperbolicTangent(double x)
        {
            return (Math.Pow(Math.E, 2 * x) - 1) / (Math.Pow(Math.E, 2 * x) + 1);
        }

        /// <summary>
        /// Производная функции активации
        /// </summary>
        /// <param name="x">Значение гиперболического тангенса в точке x</param>
        /// <returns>Значение производной гиперболического тангенса</returns>
        public static double DeriativeHyperbolicTangent(double x)
        {
            return (1 + x) * (1 - x);
        }

        /// <summary>
        /// Сохраняет сеть
        /// </summary>
        /// <param name="path">Путь файла</param>
        public void SaveNetwork(string path)
        {
            object[] saveData = new object[] { UseBiasNeurons, layers};
            var jsonData = JsonConvert.SerializeObject(saveData);
            try
            {
                File.WriteAllText(path, jsonData);
            }
            catch (FileNotFoundException)
            {
                File.Create(path);
                File.WriteAllText(path, jsonData);
            }
        }

        /// <summary>
        /// Загружает сеть
        /// </summary>
        /// <param name="path">Путь файла</param>
        public void LoadNetwork(string path)
        {
            try
            {
                var loadData = File.ReadAllText(path);
                var jsonData = JsonConvert.DeserializeObject<object[]>(loadData);
                UseBiasNeurons = (bool)jsonData[0];
                layers = jsonData[1] as Layer[];
            }
            catch
            {
                throw new Exception("Failed to load save file");
            }
        }
    }
}