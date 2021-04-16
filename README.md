# SAF Vectorizers

**SAF Vectorizers** - Плагин для SmartApp Framework, осуществляющий векторизацию (получение embedding'ов) 
текстов с помощью различных моделей

## Оглавление
   * [Установка](#Установка)
   * [Новый функционал](#Новый)
   * [Документация](#Документация)
   * [Обратная связь](#Обратная)

____

# Установка  
Перед началом установки необходимо запустить скрипт на скачивание предобученных моделей векторизаторов, 
предварительно выдайте файлу права на исполнение и ОТКЛЮЧИТЕ VPN (если используете): 
```bash
chmod u+r+x download_models.sh 
./download_models.sh
```

Установить сам плагин можно из git.

```bash
python -m pip install git+https://github.com/sberdevices/saf_vectorizers@main
```

# Новый функционал


# Документация

https://developer.sberdevices.ru/docs/ru/developer_tools/framework/

# Обратная связь

C вопросами и предложениями пишите нам по адресу developer@sberdevices.ru или вступайте 
в наш Telegram канал - [SmartApp Studio Community](https://t.me/smartapp_studio). 
