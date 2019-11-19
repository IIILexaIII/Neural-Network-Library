# Neural Network Library
Моя библиотека нейронной сети, созданная для лучшего понимания темы и решения различных проблем. Для обучения нейронной сети используется метод обратного распространения ошибки и метод градиентного спуска.

# Как пользоваться
1. Создаем экземпляр класса:
```C#
// Функция активации
NeuralNetwork.ActivationFunction function = NeuralNetwork.Sigmoid;
// Производная функции активации
NeuralNetwork.ActivationFunction deriativeFunction = NeuralNetwork.DeriativeSigmoid;
// Использовать ли нейроны смещения
bool useBiasNeurons = true;
// Количество нейронов в каждом слое
int[] neuronsCount = new int[] {2, 8, 6, 2};

NeuralNetwork NN = new NeuralNetwork(function, deriativeFunction, useBiasNeurons, neuronsCount);
```
> Библиотека нейронной сети уже содержит две функции активации (сигмоид, гиперболический тангенс), а так же их производные. При необходимости вы можете написать свою функцию активации

При создании экземпляра класса автоматически вызывается инициализатор весов, который задает всем весам значение 0. Если вам нужны случайные значения весов при инициализации используйте:
```C#
// Зерно
int seed = 2020;
// Граница значений весов при инициализации
double border = 1;

NN+.InitializationWeights(seed, border);
```
Если у вас есть уже есть датапак обученой нейросети, то используйте:
```C#
// Функция активации
NeuralNetwork.ActivationFunction function = NeuralNetwork.Sigmoid;
// Производная функции активации
NeuralNetwork.ActivationFunction deriativeFunction = NeuralNetwork.DeriativeSigmoid;
// Путь на датапак в формате Json
string path = "D:\NNData.json";

NeuralNetwork NN = new NeuralNetwork(function, deriativeFunction, path);
```

2. Теперь можно приступить к использованию нейронной сети:
```C#
// Входные значения
double[] inputValues = new int[] {0, 1};
// Выходные значения
double[] outputValues;

outputValues = NN.CalculateLayers(inputValues);
```

3. Для обучения сети нужно создать экземпляр класса NeuralNetworkTrainer:
```C#
// Сеть, которую мы будем обучать
NeuralNetwork network = NN;
// Коэффициент обучения
double learningRatio = 0.3;
// Коэффициент момента
double moment = 0.5;

NeuralNetworkTrainer NNT = new NeuralNetworkTrainer(network, learningRatio, moment);
```
Теперь, после вызова метода NeuralNetwork.CalculateLayers(...) мы должны вызвать метод NeuralNetworkTrainer.Train(...)
```C#
// Значения выходных нейронов, которые должны быть на выходе
double[] idealValues = new double[] {1, 1};

NNT.Train(idealValues);

// MSE ошибка
double MSEError = NNT.GetMSE();
```

# Ссылки
При создании исполизовались эти статьи:
1. https://habr.com/ru/post/312450/
2. https://habr.com/ru/post/313216/

Тесты нейросети:
1. https://vk.com/wall296596745_413
2. https://vk.com/wall296596745_417
