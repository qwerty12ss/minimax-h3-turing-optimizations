# Развёртывание MiniMax H3 с нуля

[English](INSTALL.md) | **Русский**

Этот рецепт устанавливает отдельный ComfyUI, FP16-загрузчик H3, PDD, TP2 и опубликованные ускорения. Для проверки используется синтетический тестовый рисунок и нейтральный текст из workflow. Модели скачиваются из публичных источников. Личные изображения, ролики, учётные данные и настройки чужого сервера не нужны.

Дополнительный вариант описан в **[рецепте Qwen TP2 + FC2 + кеш DiT](RECIPE_LUNA.ru.md)**. Он включается отдельно: **Qwen меняет и видео, и звук при том же seed**; режим только с кешем сохранил выход побитово в проверенном случае.

## Оборудование и система

- Linux x86_64; установщик рассчитан на Python **3.12**. Подходящая отправная точка — Ubuntu 24.04.
- **Две RTX 2080 Ti с 22 ГиБ VRAM каждая**, SM75, с работающим CUDA P2P в обе стороны. Обычные 2080 Ti на 11 ГиБ этим рецептом не поддерживаются.
- Ориентир по RAM — **128 ГБ**. Пять моделей занимают около **43.84 ГБ / 40.83 ГиБ**. Заложите **90 ГБ свободного места** под модели, окружение, временные файлы и результат.
- NVIDIA driver с поддержкой CUDA 13.0 и **CUDA Toolkit 13.0**, включая `nvcc`; одного драйвера или runtime из PyTorch для сборки недостаточно. Установите по [официальной инструкции NVIDIA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) и [архиву CUDA 13.0](https://developer.nvidia.com/cuda-13-0-0-download-archive). В записанной системе используется драйвер ветки 580.
- Исторические результаты получены на 2 × Xeon E5-2686 v4, 128 ГБ RAM, PCIe 3.0 ×16 на каждую карту и NV2. Полная таблица — [HARDWARE.md](HARDWARE.md). Другая топология требует проверки P2P и нового замера.

Windows напрямую не поддерживается этими CUDA build-рецептами. Для WSL2 нужна рабочая виртуализация и отдельно проверенный CUDA P2P; этот вариант не квалифицирован.

## 1. Системные пакеты и репозиторий

```bash
sudo apt update
sudo apt install -y git build-essential python3.12 python3.12-venv ffmpeg
nvcc --version
nvidia-smi

git clone https://github.com/qwerty12ss/minimax-h3-turing-optimizations.git
cd minimax-h3-turing-optimizations
export CUDA_HOME=/usr/local/cuda
python3.12 tools/setup_stack.py --root "$HOME/h3-turing-runtime"
```

Если Toolkit установлен в другом месте, задайте его каталог в `CUDA_HOME`. Установщик создаёт отдельные `ComfyUI` и `.venv` вне репозитория, получает конкретные Git-коммиты из [stack.lock.json](../configs/stack.lock.json), устанавливает зависимости с [ограничениями версий](../requirements/runtime-constraints.txt), выполняет `pip check` и собирает attention/SwiGLU для SM75. Существующие установки ComfyUI он не меняет. Повторный запуск разрешён только для созданного им каталога с теми же версиями. Каталог исходников должен оставаться на месте: пакет установлен в editable-режиме.

## 2. Модели

```bash
python3.12 tools/models.py download --root "$HOME/h3-turing-runtime"
```

Авторизация Hugging Face и токены не требуются. Скрипт скачивает пять файлов в `ComfyUI/models`, проверяет размер и полный SHA256 перед принятием каждого файла. Версии, точные имена и контрольные суммы находятся в [models.lock.json](../configs/models.lock.json).

| Назначение | Файл | Источник |
|---|---|---|
| FL2VA INT8 ConvRot | `diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/4b53a35191437de05370da548e59aeabe2a5a09a/diffusion_models) |
| Qwen3-VL text encoder | `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | [Text encoders](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/4b53a35191437de05370da548e59aeabe2a5a09a/text_encoders) |
| Video VAE FP16 | `vae/minimax_h3_video_vae_fp16.safetensors` | [VAE](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/4b53a35191437de05370da548e59aeabe2a5a09a/vae) |
| Audio VAE FP32 | `vae/minimax_h3_audio_vae_fp32.safetensors` | [VAE](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/4b53a35191437de05370da548e59aeabe2a5a09a/vae) |
| PDD FL2VA, 8 шагов | `pdd_acc/MiniMax-H3-FL2VA-Acc-8Step.safetensors` | [alibaba-pai/MiniMax-H3-Acc-LoRAs](https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs/tree/335001fb9e5455d68a0caa18ec2e319072150328) |

Лицензии и условия моделей приведены в их исходных карточках: [MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3), [PDD](https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs). Веса не входят в этот Git-репозиторий.

Если эти же публичные модели уже скачаны, можно использовать отдельный каталог-кэш:

```bash
python3.12 tools/models.py link --root "$HOME/h3-turing-runtime" --from-models /path/to/existing/ComfyUI/models
```

Это создаёт ссылки после полной проверки SHA256; исходные файлы не изменяются. Не подставляйте PDD-baked checkpoint: готовые workflow ожидают исходный FL2VA INT8 и отдельный адаптер PDD. Для повторной проверки используйте `tools/models.py verify --root "$HOME/h3-turing-runtime"`.

## 3. Запуск ComfyUI

```bash
python3.12 tools/launch.py --root "$HOME/h3-turing-runtime" --gpus 0,1 --port 8188
```

`--gpus` — физические индексы из `nvidia-smi`. Внутри процесса выбранная пара становится `cuda:0` и `cuda:1`. Запуск проверяет SM75, объём памяти и P2P. ComfyUI слушает только `127.0.0.1`; открыть интерфейс можно по `http://127.0.0.1:8188`. На удалённой машине используйте свой SSH tunnel. Если порт занят другой установкой, выберите другой, например `--port 8197`, и передайте тот же порт клиенту ниже.

Launcher разрешает физические индексы независимо от порядка CUDA на машинах с разными GPU. Он задаёт обычный CUDA caching allocator (`--disable-cuda-malloc`), отключает dynamic VRAM и кэш графа. Эти параметры входят в проверенный рецепт: при сквозной проверке асинхронный allocator исчерпал память, тогда как обычный выполнил короткий профиль.

Для новой проверки освободите выбранные GPU от других задач. Работа под посторонней нагрузкой не даёт сопоставимого замера скорости. Не включайте `python -O`: исходные вычислительные контракты используют assertions.

## 4. Готовая проверка видео со звуком

В другом терминале из каталога репозитория:

```bash
# 640×352, 124 кадра, 8 шагов, последовательный VAE
python3.12 tools/run_workflow.py workflows/h3_turing_smoke_api.json

# 960×512, 192 кадра, 24 fps, 8 шагов, temporal hybrid VAE
python3.12 tools/run_workflow.py workflows/h3_turing_8s_api.json
```

Для другого порта добавьте `--url http://127.0.0.1:8197`. Клиент ждёт завершения, сообщает ошибку узла и проверяет наличие численного отчёта о конечных кадрах и звуке. Результат сохраняется в `h3-turing-runtime/ComfyUI/output/video`. Флаг `--report /path/to/local-receipt.json` сохраняет небольшой локальный отчёт без промпта и идентификаторов машины.

API JSON предназначен для этого клиента; для визуального редактора используйте файлы `*_gui.json` из [workflows](../workflows). Не отправляйте API JSON в редактор как обычный GUI workflow.

## Параметры и собственный ввод

Сначала запустите готовую проверку. Затем замените `H3TuringTestImage` узлом `LoadImage` и подайте собственное изображение в `MiniMaxH3ImageToVideo.first_frame`; текст меняется в `prompt` этого узла. Для API нужно загрузить изображение в свой ComfyUI и изменить JSON соответственно. Не коммитьте изображения, вывод или локальные журналы.

Сохраняйте связку: FL2VA → SigmaShift **12/3** → PDD FL2VA **nfe=8**, strength **1/1** → Activation Chunk → H3TuringApply → BasicGuider **CFG 1** → **Euler**, причём `sigmas` идут из PDD. Профиль attention: CK INT8 на шагах 1, 2, 7, 8; Sol all-INT8 на шагах 3–6, `tau=1`. QKV/MLP chunks **8192**. У PDD свой обученный график; обычный scheduler или случайная замена числа шагов не эквивалентны этому workflow.

`preset=baseline` в H3TuringApply оставляет исходный TP2 central block с Half A/B PDD, сохраняя гибридное attention. `optimized` добавляет общую подготовку QKV/FC1, точные scale/add и SUM, norm/mod, direct-output RoPE и joint SwiGLU/ConvRot. Эти режимы не заменяют квантованные веса на полностью BF16-модель.

Temporal hybrid VAE квалифицируется только для перечисленных в `vae/spatial.py` форм, включая **960×512, 192 кадра**. Для других размеров используйте `mode=serial` и отдельно проверяйте память/результат. Кадры H3 лежат на сетке **17k+5**: 124 и 192 подходят. Готовые профили не обещают поддержку произвольной длины, разрешения, иной архитектуры GPU или одновременного выполнения двух TP2 моделей.

## Проверки и типичные ошибки

```bash
"$HOME/h3-turing-runtime/.venv/bin/python" -m pytest -q
CUDA_VISIBLE_DEVICES=0,1 "$HOME/h3-turing-runtime/.venv/bin/python" benchmarks/attention_smoke.py
CUDA_VISIBLE_DEVICES=0,1 H3_RUN_GPU_TESTS=1 "$HOME/h3-turing-runtime/.venv/bin/python" -m pytest -q -m gpu
```

- `Missing ComfyUI nodes`: смотрите ошибки импорта при старте, используйте установщик и зафиксированные коммиты.
- `SHA256 mismatch`: не используйте файл; перепроверьте источник и докачку. Скрипт не заменяет существующий неверный файл автоматически.
- `Original ... source changed`: версия CK/ядра отличается от зафиксированной. Создайте чистое окружение; простая замена ожидаемого хеша не доказывает корректность.
- `nvcc missing` или ошибка сборки: проверьте Toolkit 13.0, `CUDA_HOME`, `c++` и свободное место.
- Нет P2P: проверьте топологию и драйвер. Этот TP2 рецепт требует доступа в обе стороны.
- OOM: проверьте **22 ГиБ на каждой карте**, чужие процессы и соответствие готовому профилю. Снижение размера smoke-проверки не делает stock 11 ГиБ поддерживаемым вариантом.
- `density=nan` в строке асинхронной статистики attention означает отложенное чтение этой метрики. Конечность самих кадров/звука проверяется отдельным узлом, а статистика маршрутов проверяется перед сохранением результата.

Свежая функциональная проверка и её ограничения публикуются отдельно от исторических замеров в [отчёте о развёртывании](DEPLOYMENT_VALIDATION.ru.md). Успех генерации и конечность чисел не являются доказательством побитового равенства всех вариантов или художественного качества видео.
