# SAF Vectorizers

**SAF Vectorizers** - Плагин для SmartApp Framework, осуществляющий векторизацию (получение embedding'ов) 
текстов с помощью различных моделей:    
- **SBERT** (SentenceBERT), предобученная русскоязычная модель от SberDevices, которая доступна в 
open source (подробнее про нее можно почитать здесь https://habr.com/en/company/sberdevices/blog/527576/);  
- **USE** (Universal Sentence Encoder), предобученная мультиязыковая модель (подробности можно 
найти здесь https://tfhub.dev/google/universal-sentence-encoder/1);  
- **FastText**, предобученная русскоязычная модель, взятая с https://fasttext.cc/docs/en/crawl-vectors.html;   
- **Word2Vec**, предобученная русскоязычная модель, взятая с http://vectors.nlpl.eu/repository.

## Оглавление
   * [Установка](#Установка)
   * [Новый функционал](#Новый)
   * [Подключение плагина](#Подключение)
   * [Документация](#Документация)
   * [Обратная связь](#Обратная)

____

# Установка  

Перед началом установки необходимо запустить скрипт на скачивание предобученных моделей векторизаторов 
(в репозитории их нет т.к все модели тяжелые), 
предварительно выдайте скрипту права на исполнение и отключите VPN (если используете): 
```bash
chmod u+r+x download_models.sh 
./download_models.sh
```
У вас должна появиться директория `static` в saf_vectorizers, там будут храниться файлы моделей, 
финальный размер директории будет около `16 ГБ`.   

Процесс скачивания моделей не быстрый и занимает какое-то время, в логах консоли можно увидеть какая 
именно модель сейчас скачивается.   

Затем следует установить сам плагин:
```bash
pip install -e .
```
Рекомендуется устанавливать именно таким образом, а не через git т.к необходимо включение файлов 
из директоии `static` (см. файл MANIFEST.in), т.е активируете env, куда у вас уже установлен smart_app_framework, или 
создаете новый env, затем переходите с склонированный репозиторий saf_vectorizers (main ветка) и запускаете 
`pip install -e .`

Проверить, что все установилось успешно в ваш env:   
```bash
from core.text_preprocessing.preprocessing_result import TextPreprocessingResult
from saf_vectorizers import SBERTVectorizer 

vectorizer=SBERTVectorizer()

test_text=TextPreprocessingResult({"original_text": "хочу узнать прогноз погоды на завтра в москве"})

res_vector = vectorizer.vectorize(test_text)

print(res_vector)
print(res_vector.shape)
```

# Новый функционал

# Подключение плагина

Чтобы подключить плагин, добавьте его имя в переменную PLUGINS в app_config вашего смартаппа:  
`PLUGINS = ["saf_vectorizers"]`

# Документация

https://developer.sberdevices.ru/docs/ru/developer_tools/framework/

# Обратная связь

C вопросами и предложениями пишите нам по адресу developer@sberdevices.ru или вступайте 
в наш Telegram канал - [SmartApp Studio Community](https://t.me/smartapp_studio). 
