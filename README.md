# SAF Vectorizers

**SAF Vectorizers** - Плагин для SmartApp Framework, осуществляющий векторизацию (получение embedding'ов) 
текстов с помощью различных моделей:    

- **SBERT** (SentenceBERT) предобученная русскоязычная модель от [SberDevices](https://sberdevices.ru), 
которая доступна в open source (подробнее про нее можно почитать [в статье на habr](
https://habr.com/en/company/sberdevices/blog/527576/)).
  
- **USE** (Universal Sentence Encoder) предобученная мультиязыковая модель (подробности про модель можно 
найти на [TensorFlow Hub](https://tfhub.dev/google/universal-sentence-encoder/1)). Модель 
распространяется под лицензией [Apache-2.0](https://opensource.org/licenses/Apache-2.0) и 
используется в оригинальном виде, без каких-либо изменений. 

- **FastText** предобученная русскоязычная модель, распространяется на условиях лицензии
 [Creative Commons Attribution-Share-Alike License 3.0](https://creativecommons.org/licenses/by-sa/3.0/). 
 Модель скачивается с официального сайта [FastText](https://fasttext.cc/docs/en/crawl-vectors.html) и 
 используется в оригинальном виде, без каких-либо изменений.   
 Авторами модели являются:
 ```
@inproceedings{grave2018learning,
  title={Learning Word Vectors for 157 Languages},
  author={Grave, Edouard and Bojanowski, Piotr and Gupta, Prakhar and Joulin, Armand and Mikolov, Tomas},
  booktitle={Proceedings of the International Conference on Language Resources and Evaluation (LREC 2018)},
  year={2018}
}
```   

- **Word2Vec** предобученная русскоязычная модель, распространяется на условиях лицензии
 [Creative Commons Attribution (CC-BY)](https://creativecommons.org/licenses/by/4.0/deed.ru). 
 Модель скачивается с официального сайта [NLPL word embeddings repository](http://vectors.nlpl.eu/repository/) и 
 используется в оригинальном виде, без каких-либо изменений.   
 Авторами модели являются [Language Technology Group at the University of Oslo](
 https://www.mn.uio.no/ifi/english/research/groups/ltg/).

*Названия типов моделей (используются как аргумент для скрипта на скачивание моделей, а также в конфигах 
классификаторов в поле "vectorizer")*: `sbert`, `use`, `fasttext`, `word2vec`

## Оглавление
   * [Установка](#Установка)
   * [Новый функционал](#Новый_функционал)
   * [Подключение плагина](#Подключение)
   * [Документация](#Документация)
   * [Обратная связь](#Обратная)

____

# Установка  

Перед началом установки рекомендуется запустить скрипт на скачивание предобученных моделей векторизаторов 
(в репозитории их нет т.к все модели тяжелые), предварительно выдайте скрипту права на исполнение 
и отключите VPN (если используете). 

В качестве аргументов скрипт принимает названия моделей векторизаторов, 
которые вы хотите скачать и использовать. Если аргумент `all`, то скачиваются все модели. Если, например, хотите
скачать и использовать только sbert, то замените `all` на `sbert`. Если нужны только use и fasttext, то вместо `all` 
пропишите `use fasttext` и т.д.

Но обратите внимание, что не обязательно запускать отдельно скрипт на скачивание моделей, т.к он по умолчанию уже 
запускается в `setup.py`. Если не хотите качать все модели, то зайдите в `setup.py` и замените `all` на другое значение.

Команда запуска скрипта на скачивание моделей:
```bash
chmod u+r+x download_models.sh 
./download_models.sh all
```

У вас должна появиться директория `static` в saf_vectorizers, там будут храниться файлы моделей, 
финальный размер директории, если вы скачаете все модели, будет около `16 ГБ`.   

Процесс скачивания моделей не быстрый и занимает какое-то время, в логах консоли можно увидеть какая 
именно модель сейчас скачивается.   

Команда установки плагина:
```bash
pip install -e .
```

Рекомендуется устанавливать именно таким образом, а не через git т.к необходимо включение файлов 
из директоии `static` (см. файл MANIFEST.in), т.е активируете env, куда у вас уже установлен smart_app_framework, или 
создаете новый env, затем переходите в склонированный репозиторий saf_vectorizers (main ветка) и запускаете 
`pip install -e .`

Проверить, что все установилось успешно в ваш env можно так:   
```bash
from core.text_preprocessing.preprocessing_result import TextPreprocessingResult
from saf_vectorizers import SBERTVectorizer 

vectorizer=SBERTVectorizer()

test_text=TextPreprocessingResult({"original_text": "хочу узнать прогноз погоды на завтра в москве"})

res_vector=vectorizer.vectorize(test_text)

print(res_vector)
print(res_vector.shape)
```

# Новый функционал

Плагин предоставляет следующие сущности:
- `class FastTextVectorizer`  
- `class SBERTVectorizer`  
- `class USEVectorizer`  
- `class Word2VecVectorizer`  

Каждый из этих классов является векторизатором, который вы можете использовать при обучение своих 
классификационных моделей, а также во время инференса, чтобы модели на вход приходило уже векторное 
представление текста. Чтобы получить векторное представление текста вам нужно вызвать у векторизатора 
метод `vectorize`. Он принимает на вход объект `TextPreprocessingResult` и возвращает вектор как NumPy массив:
```python
def vectorize(self, text_preprocessing_result: TextPreprocessingResult) -> np.ndarray:
```

Пример объекта `TextPreprocessingResult` можно найти здесь: 
https://github.com/sberdevices/saf_vectorizers/blob/main/saf_vectorizers/check_vectorizers.py

# Подключение плагина

Чтобы подключить плагин, добавьте его имя в переменную `PLUGINS` в app_config вашего смартаппа:  
`PLUGINS = ["saf_vectorizers"]`  

В конфигурации классификатора, модель которого должна принимать на вход уже векторизированную реплику, 
следует добавить поле `"vectorizer"` с одним из значений (`sbert`, `use`, `fasttext`, `word2vec`) 
типа модели векторизации, та же что использовалась при обучение модели:
```json
{
    "type": "scikit",
    "threshold": 0.7,
    "path": "pretrained_model.pkl",
    "intents": ["intent_1", "intent_2" ... "intent_n"],
    "vectorizer": "sbert"
}
```

# Документация

[Официальная документация](https://developer.sberdevices.ru/docs/ru/developer_tools/framework/overview.md)

# Обратная связь

C вопросами и предложениями пишите нам по адресу developer@sberdevices.ru или вступайте 
в наш Telegram канал - [SmartApp Studio Community](https://t.me/smartapp_studio). 
