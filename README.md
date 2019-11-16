Не доделано
# Neural Network Library
Для обучения нейронной сети используется метод обратного распространения ошибки и метод градиентного спуска.

# Как пользоваться
1. Создаем экземпляр класса:
```C#
// Функция активации
ActivationFunction function = NeuralNetwork.Sigmoid;
// Производная функции активации
ActivationFunction deriativeFunction = NeuralNetwork.DeriativeSigmoid;
// Использовать ли нейроны смещения
bool useBiasNeurons = true;
// Количество нейронов в каждом слое
int[] neuronsCount = new int[] {2, 8, 6, 2};

NeuralNetwork NN = new NeuralNetwork(function, deriativeFunction, useBiasNeurons, neuronsCount);
```
При создании экземпляра класса автоматически вызывается инициализатор весов, который задает всем весам значение 0. Если вам нужны случайные значения весов при инициализации используйте:
```C#
// Зерно
int seed = 2020;
// Граница значений весов при инициализации
double border = 1;

NeuralNetwork.InitializationWeights(seed, border);
```
Если у вас есть уже есть датапак обученой нейросети, то используйте:
```C#
// Функция активации
ActivationFunction function = NeuralNetwork.Sigmoid;
// Производная функции активации
ActivationFunction deriativeFunction = NeuralNetwork.DeriativeSigmoid;
// Путь на датапак в формате Json
string path = "D:\NNData.json";

NeuralNetwork NN = new NeuralNetwork(ActivationFunction function, ActivationFunction deriativeFunction, string path);
```

2. Теперь можно приступить к использованию нейронной сети:
```C#
// Входные значения
double[] inputValues = new int[] {0, 1};
// Выходные значения
double[] outputValues;

outputValues = NN.CalculateLayers(inputValues)
```

# Ссылки
При создании исполизовались эти статьи:
1. https://habr.com/ru/post/312450/
2. https://habr.com/ru/post/313216/

Тесты нейросети:
1. https://vk.com/wall296596745_413
2. https://vk.com/wall296596745_417
