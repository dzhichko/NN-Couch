# Для запуска Backend
Требуется создать два терминала (перед этим подняв mongodb с docker-compose)

**Первый терминал**
```commandline
cd Backend/
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
Это запустит бэк

**Второй терминал**
```commandline
cd Backend/
python -m http.server 3000
```
Это запустит фронт. При переходе на http://localhost:3000 нужно выбрать файл app.html и всё будет работать