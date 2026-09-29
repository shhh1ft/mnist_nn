# mnist_nn — нейросеть с нуля на NumPy

Реализация, обучение и оценка нейронной сети для распознавания рукописных
цифр MNIST **без использования ML-фреймворков** (PyTorch, TensorFlow,
Keras, JAX). NumPy применяется исключительно для матричных операций.
Вся логика слоёв, прямого и обратного прохода, оптимизаторов и функций
потерь написана вручную.

Проект разработан по ТЗ за 8 недель и включает:
полносвязную сеть (MLP), свёрточную сеть (CNN) с нуля,
метрики и визуализацию, сравнение моделей, бенчмарки,
Docker-упаковку, тесты (110+), полный пайплайн одной командой
и **GUI для рисования цифры мышкой**.

---

## Содержание

- [Возможности](#возможности)
- [Быстрый старт](#быстрый-старт)
- [Интерактивное распознавание (GUI)](#интерактивное-распознавание-gui)
- [Структура проекта](#структура-проекта)
- [Воспроизведение результатов](#воспроизведение-результатов)
- [Архитектуры моделей](#архитектуры-моделей)
- [Математические выкладки](#математические-выкладки)
- [Бенчмарки](#бенчмарки)
- [Docker](#docker)
- [Тесты](#тесты)
- [Приёмо-сдаточные испытания](#приёмо-сдаточные-испытания)
- [Требования](#требования)
- [Лицензия](#лицензия)

---

## Возможности

**Слои и активации**
- `Dense` — полносвязный слой с инициализацией He / Xavier;
- `ReLU`, `Dropout` (inverted, train/eval режимы);
- `Conv2D` — свёртка через **im2col/col2im** (быстро, полностью векторизовано);
- `MaxPool2D` — с сохранением индексов argmax для backward;
- `Flatten`.

**Loss и оптимизация**
- `SoftmaxCrossEntropy` — совместная реализация, численно устойчивая,
  градиент `(ŷ − y) / N`;
- `SGD` с momentum и L2 (weight decay);
- `Adam` с коррекцией смещения;
- **Gradient clipping** по глобальной L2-норме;
- Планировщики LR: `ConstantLR`, `StepDecay`, `CosineAnnealing`.

**Обучение**
- мини-батчи с перемешиванием;
- early stopping по validation loss + снапшот лучших весов;
- логирование истории (loss, accuracy, lr) в `reports/*_history.json`;
- сохранение/загрузка весов в `.npz`.

**Метрики и визуализация**
- accuracy, precision / recall / F1 (per-class, macro, weighted);
- confusion matrix 10×10;
- графики loss / accuracy / lr;
- сетка ошибочных предсказаний с подписью true / pred / confidence.

**Сравнение и анализ**
- `src.compare` — сравнение нескольких моделей по accuracy, loss, F1,
  числу параметров, времени inference;
- `src.analyze_errors` — анализ расхождений между двумя моделями:
  где A ошиблась, а B угадала, и наоборот;
- `src.report` — сборка итогового `reports/final_report.md`.

**Бенчмарк**
- `benchmarks.benchmark` — замер inference на CPU для batch = 1 / 32 / 128;
  выводит p50, p95, mean, std и сохраняет `results.json` + `benchmark.csv`.

**GUI**
- `src.gui` — оконное приложение на **Tkinter** (без новых зависимостей):
  рисуешь цифру мышкой — модель сразу выдаёт предсказание и распределение
  вероятностей.

**Инфраструктура**
- `scripts.run_all` — полный пайплайн одной командой:
  тесты → данные → обучение → оценка → сравнение → бенчмарк → отчёт;
- `Dockerfile` и `docker-compose.yml` — воспроизводимая среда;
- GitHub Actions CI на Python 3.10 / 3.11 / 3.12.

---

## Быстрый старт

```bash
python -m venv .venv
source .venv/bin/activate            # Linux / macOS
# .venv\Scripts\activate             # Windows
pip install -r requirements.txt
```

### Обучение MLP

```bash
python -m src.train --config configs/mlp_baseline.json
```

### Обучение CNN

```bash
python -m src.train --config configs/cnn_baseline.json
```

### Оценка на тесте

```bash
python -m src.evaluate \
    --model models/cnn_best.npz \
    --config configs/cnn_baseline.json \
    --split test --tag cnn
```

### Сравнение нескольких моделей

```bash
python -m src.compare \
    --entry mlp=models/mlp_best.npz,configs/mlp_baseline.json \
    --entry cnn=models/cnn_best.npz,configs/cnn_baseline.json \
    --tag comparison
```

### Полный пайплайн одной командой

```bash
python -m scripts.run_all
```

Что произойдёт: прогон тестов → скачивание MNIST → подготовка данных →
обучение MLP и CNN → оценка → сравнение → анализ ошибок → бенчмарк →
итоговый отчёт в `reports/final_report.md`.

---

## Интерактивное распознавание (GUI)

Оконное приложение на Tkinter (входит в стандартную библиотеку Python).

### Запуск с автоопределением модели

```bash
python -m src.gui
```

Скрипт сам найдёт обученные веса в `models/*.npz`, подберёт соответствующий
конфиг в `configs/` и отдаст приоритет CNN, если он есть.

### Запуск с явным указанием модели

```bash
python -m src.gui \
    --model models/cnn_best.npz \
    --config configs/cnn_baseline.json
```

### Отключение MNIST-предобработки (для сравнения)

```bash
python -m src.gui --no-preprocess
```

### Как пользоваться

1. Рисуешь цифру чёрным по белому холсту 280×280.
2. На отпускании левой кнопки мыши модель **сразу** выдаёт предсказание.
3. Справа видишь:
   - крупную цифру — предсказанный класс;
   - уверенность в процентах;
   - распределение вероятностей 0–9 в виде полос.
4. **Очистить** — стереть холст; **Распознать** — пересчитать.
5. Чекбокс **«MNIST-предобработка»** — включает/выключает crop + центрирование.

### Про предобработку

MNIST создавался по стандарту: каждая цифра обрезана по bounding box,
масштабирована так, что большая сторона = 20 px, и отцентрирована
по центру масс в поле 28×28. Модель обучалась именно на таких данных.

Если подать «сырой» рисунок с холста 280×280 напрямую — точность падает
до 50–70%, потому что цифра окажется смещённой и слишком крупной.
Модуль `src/preprocess.py` повторяет стандартный пайплайн MNIST:
crop → resize (билинейная интерполяция) → центрирование по центру масс.
После этого точность на нарисованных от руки цифрах — **96–99%** для CNN.

Если `tkinter` отсутствует на Linux:

```bash
sudo apt install python3-tk
```

На Windows и macOS `tkinter` идёт в комплекте с CPython.

---

## Структура проекта

```text
mnist_nn/
├── src/
│   ├── data_loader.py      # IDX-парсер, нормализация, one-hot, split, батчи, кэш
│   ├── layers.py           # Dense, ReLU, Dropout
│   ├── conv.py             # im2col/col2im, Conv2D, MaxPool2D, Flatten
│   ├── losses.py           # SoftmaxCrossEntropy
│   ├── optimizers.py       # SGD, Adam, gradient clipping
│   ├── schedulers.py       # ConstantLR, StepDecay, CosineAnnealing
│   ├── model.py            # Sequential, build_model_from_config, save/load
│   ├── trainer.py          # цикл обучения, early stopping, снапшоты
│   ├── metrics.py          # accuracy, P/R/F1, confusion matrix
│   ├── visualize.py        # графики, CM, ошибочные примеры, сравнение
│   ├── preprocess.py       # MNIST-предобработка (crop + center)
│   ├── train.py            # CLI обучения
│   ├── evaluate.py         # CLI оценки
│   ├── predict.py          # CLI предсказания (одиночное изображение / директория)
│   ├── compare.py          # сравнение нескольких моделей
│   ├── analyze_errors.py   # анализ расхождений между двумя моделями
│   ├── report.py           # генератор итогового отчёта
│   ├── gui.py              # Tkinter GUI: рисуем — распознаём
│   └── utils.py            # seed, директории, загрузка MNIST, softmax
├── benchmarks/
│   └── benchmark.py        # замер inference на CPU (p50, p95)
├── scripts/
│   ├── download_mnist.py   # скачать данные
│   ├── prepare_data.py     # подготовить данные
│   └── run_all.py          # полный пайплайн одной командой
├── configs/
│   ├── mlp_baseline.json
│   ├── mlp_adam_dropout.json
│   ├── mlp_adam_step.json
│   ├── mlp_sgd_cosine.json
│   ├── cnn_baseline.json
│   ├── cnn_deep.json
│   └── cnn_wide.json
├── tests/                  # pytest (110+ тестов)
│   ├── conftest.py
│   ├── test_data_loader.py
│   ├── test_layers.py
│   ├── test_conv.py
│   ├── test_optimizers.py
│   ├── test_schedulers.py
│   ├── test_metrics.py
│   ├── test_visualize.py
│   ├── test_gradient_check.py
│   ├── test_compare.py
│   ├── test_preprocess.py
│   └── test_release.py
├── docs/
│   └── acceptance.md       # протокол приёмо-сдаточных испытаний
├── data/                   # raw + processed (.gitignore)
├── models/                 # чекпоинты (.gitignore)
├── reports/                # метрики, графики, отчёт
├── benchmarks/             # результаты бенчмарков
├── .github/workflows/ci.yml
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Воспроизведение результатов

Все эксперименты воспроизводимы благодаря фиксации seed.

- seed задаётся в JSON-конфиге (`"seed": 42`);
- можно переопределить флагом: `--seed 123`;
- seed влияет на:
  - инициализацию весов (Dense, Conv2D);
  - перемешивание батчей;
  - dropout-маску;
  - стратифицированное разбиение train/val.

Проверка воспроизводимости:

```bash
pytest tests/test_release.py::test_seed_reproducibility_training -v
```

### Полный прогон с нуля

```bash
# 1. Скачать и подготовить данные
python -m scripts.download_mnist
python -m scripts.prepare_data

# 2. Обучить
python -m src.train --config configs/mlp_baseline.json
python -m src.train --config configs/cnn_baseline.json

# 3. Оценить
python -m src.evaluate --model models/mlp_best.npz \
    --config configs/mlp_baseline.json --tag mlp
python -m src.evaluate --model models/cnn_best.npz \
    --config configs/cnn_baseline.json --tag cnn

# 4. Сравнить
python -m src.compare \
    --entry mlp=models/mlp_best.npz,configs/mlp_baseline.json \
    --entry cnn=models/cnn_best.npz,configs/cnn_baseline.json \
    --tag comparison

# 5. Проанализировать расхождения
python -m src.analyze_errors \
    --a mlp=models/mlp_best.npz,configs/mlp_baseline.json \
    --b cnn=models/cnn_best.npz,configs/cnn_baseline.json \
    --tag mlp_vs_cnn

# 6. Бенчмарк
python -m benchmarks.benchmark \
    --entry mlp=models/mlp_best.npz,configs/mlp_baseline.json \
    --entry cnn=models/cnn_best.npz,configs/cnn_baseline.json

# 7. Итоговый отчёт
python -m src.report --tag final
```

Все артефакты: `reports/` и `benchmarks/`.

---

## Архитектуры моделей

### MLP baseline (`configs/mlp_baseline.json`)

```text
Вход (784) → Dense (128) → ReLU → Dense (10) → Softmax
```

Целевая точность: **≥ 97,5%**.

### MLP + Dropout + Adam (`configs/mlp_adam_dropout.json`)

```text
Вход (784)
  → Dense (256) → ReLU → Dropout (0.3)
  → Dense (128) → ReLU → Dropout (0.2)
  → Dense (10) → Softmax
```

Целевая точность: **~ 98,0%**.

### CNN baseline (`configs/cnn_baseline.json`)

```text
Вход (1, 28, 28)
  → Conv2D (8, 3×3, valid) → ReLU → MaxPool2D (2×2)
  → Conv2D (16, 3×3, valid) → ReLU → MaxPool2D (2×2)
  → Flatten (400)
  → Dense (128) → ReLU → Dense (10) → Softmax
```

Целевая точность: **≥ 99,0%**.

### CNN deep (`configs/cnn_deep.json`)

```text
Вход (1, 28, 28)
  → Conv2D (16, 3×3, same) → ReLU → Conv2D (16, 3×3, same) → ReLU
  → MaxPool2D (2×2)
  → Conv2D (32, 3×3, same) → ReLU → Conv2D (32, 3×3, same) → ReLU
  → MaxPool2D (2×2)
  → Flatten (1568)
  → Dense (128) → ReLU → Dropout (0.5) → Dense (10) → Softmax
```

Целевая точность: **~ 99,2%**.

### CNN wide (`configs/cnn_wide.json`)

```text
Вход (1, 28, 28)
  → Conv2D (32, 3×3, same) → ReLU → MaxPool2D (2×2)
  → Conv2D (64, 3×3, same) → ReLU → MaxPool2D (2×2)
  → Flatten (3136)
  → Dense (256) → ReLU → Dropout (0.5) → Dense (10) → Softmax
```

Целевая точность: **~ 99,2%**.

---

## Математические выкладки

### Dense

```
Forward:  Z = X · W + b
Backward: dW = Xᵀ · dZ
          db = Σ dZ по батчу
          dX = dZ · Wᵀ
```

### ReLU

```
Forward:  A = max(0, Z)
Backward: dZ = dA · (Z > 0)
```

### Dropout (inverted)

```
Train:  mask ~ Bernoulli(keep)
        A = X · mask / keep
Eval:   A = X
```

### Softmax + Cross-Entropy

```
Forward:  ŷ = softmax(Z)              (с вычитанием max для устойчивости)
          L = −Σ y · log(ŷ) / N
Backward: dZ = (ŷ − y) / N
```

### Conv2D через im2col

```
Forward:
  cols = im2col(X)                    # (N, C·kH·kW, H_out·W_out)
  out  = W_col @ cols + b             # W_col: (C_out, C·kH·kW)

Backward:
  dW_col = dOut_flat @ colsᵀ
  db     = Σ dOut_flat
  dCols  = W_colᵀ @ dOut_flat
  dX     = col2im(dCols)
```

### MaxPool2D

```
Forward:  out[n,c,i,j] = max(patch)   (индексы argmax кэшируются)
Backward: градиент идёт только в позицию argmax каждого патча
```

### SGD с momentum и weight decay

```
g ← ∂L/∂W + wd · W
v ← μ · v − lr · g
W ← W + v
```

### Adam

```
m ← β₁ · m + (1 − β₁) · g
v ← β₂ · v + (1 − β₂) · g²
m̂ = m / (1 − β₁ᵗ);   v̂ = v / (1 − β₂ᵗ)
W ← W − lr · m̂ / (√v̂ + ε)
```

### Cosine annealing LR

```
lr(t) = min_lr + 0.5 · (base_lr − min_lr) · (1 + cos(π · t / t_max))
```

### Gradient clipping по глобальной норме

```
‖g‖ = sqrt(Σ ‖gᵢ‖²)
если ‖g‖ > max_norm:  gᵢ ← gᵢ · max_norm / ‖g‖
```

---

## Бенчмарки

```bash
python -m benchmarks.benchmark \
    --entry mlp=models/mlp_best.npz,configs/mlp_baseline.json \
    --entry cnn=models/cnn_best.npz,configs/cnn_baseline.json \
    --runs 30 --warmup 5 --batches 1 32 128
```

Вывод (пример):

```text
[mlp] batch=   1 mean=   0.05 ms  p50=   0.05  p95=   0.07
[mlp] batch=  32 mean=   0.39 ms  p50=   0.38  p95=   0.45
[mlp] batch= 128 mean=   1.42 ms  p50=   1.40  p95=   1.60
[cnn] batch=   1 mean=   5.10 ms  p50=   5.05  p95=   5.40
[cnn] batch=  32 mean=   9.80 ms  p50=   9.75  p95=  10.20
[cnn] batch= 128 mean=  28.50 ms  p50=  28.30  p95=  29.10
```

Результаты сохраняются в `benchmarks/results.json` и `benchmarks/benchmark.csv`.

---

## Docker

```bash
# Сборка
docker compose build

# Полный пайплайн в контейнере
docker compose run --rm mnist python -m scripts.run_all

# Или конкретная команда
docker compose run --rm mnist python -m src.gui
```

Артефакты монтируются наружу: `data/`, `models/`, `reports/`, `benchmarks/`.

> Запуск GUI из контейнера требует проброса X11 (Linux) или X-сервера
> (Windows / macOS) — это выходит за рамки базового сценария. Проще
> запускать `src.gui` локально после обучения моделей.

---

## Тесты

```bash
pytest tests/ -v
```

Ожидаемо: **110+ passed**.

| Файл | Что проверяет |
|---|---|
| `test_data_loader.py` | IDX-парсинг, нормализация, one-hot, split, батчи, кэш |
| `test_layers.py` | Dense, ReLU, Dropout |
| `test_conv.py` | im2col / col2im, Conv2D, MaxPool2D, Flatten, gradient check |
| `test_optimizers.py` | SGD, Adam, gradient clipping |
| `test_schedulers.py` | ConstantLR, StepDecay, CosineAnnealing |
| `test_metrics.py` | accuracy, P / R / F1, CM, classification_report |
| `test_visualize.py` | сохранение PNG без ошибок |
| `test_gradient_check.py` | численная проверка градиентов (Dense, Softmax-CE, MLP) |
| `test_compare.py` | сравнение моделей, парсинг аргументов |
| `test_preprocess.py` | MNIST-предобработка (crop, resize, центрирование) |
| `test_release.py` | save/load, воспроизводимость seed, отчёт, smoke |

Быстрый прогон только smoke-тестов:

```bash
pytest tests/test_release.py -q
```

---

## Приёмо-сдаточные испытания

Протокол — в `docs/acceptance.md`. В нём:

- объём поставки;
- таблица функциональных требований с указанием способа проверки;
- таблица нефункциональных требований (точность, время, покрытие);
- команды воспроизведения;
- поля для замечаний и подписей.

---

## Требования

- **Python:** 3.10+ (проверено на 3.10, 3.11, 3.12)
- **ОС:** Windows, Linux, macOS
- **Железо:** любой CPU; GPU не требуется, обучение на CPU
- **Пакеты:** `numpy`, `matplotlib`, `pytest`
- **Стандартная библиотека:** `tkinter` (для GUI)
- **Запрещены:** `torch`, `tensorflow`, `keras`, `jax`, `sklearn`
  и другие ML-фреймворки

Установка:

```bash
pip install -r requirements.txt
```

---

## Лицензия

Учебно-прикладной проект. Передаётся заказчику по договору отчуждения
исключительных или неисключительных прав. Исходные коды, документация,
обученные веса и настроенная инфраструктура включены в объём поставки.

---

## Благодарности

- Yann LeCun и коллегам — за датасет MNIST;
- сообществу NumPy — за стабильный и быстрый инструмент матричных операций;
- разработчикам Tkinter — за возможность сделать GUI без внешних зависимостей.