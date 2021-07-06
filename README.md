# python-flask-docker
Итоговый проект курса "Машинное обучение в бизнесе"

Стек:

ML: sklearn, pandas, numpy
API: flask
Данные: с kaggle - https://www.kaggle.com/rounakbanik/the-movies-dataset

Задача: Построить гибридную рекомендательную систему


keywords.csv: Contains the movie plot keywords for our MovieLens movies. Available in the form of a stringified JSON Object.

credits.csv: Consists of Cast and Crew Information for all our movies. Available in the form of a stringified JSON Object.

links.csv: The file that contains the TMDB and IMDB IDs of all the movies featured in the Full MovieLens dataset.

links_small.csv: Contains the TMDB and IMDB IDs of a small subset of 9,000 movies of the Full Dataset.

ratings_small.csv: The subset of 100,000 ratings from 700 users on 9,000 movies.

movies_metadata.csv - Главный файл с метадатой. Состоить из информации о 45,000 фильмаов использованный MovieLens датасете. 
Используемые признаки:
 - adult (bool)
 - belongs_to_collection
 - budget(int)
 - genres(text)
 - homepage(text)
 - id(int)
 - original_language(text)
 - original_title(text)
 - overview(text)
 - poster_path(text)
 - release_date(text)
 - revenue(int)
 - title(text)
 - vote_average(int)
 - vote_count(int)

credits.csv Актерский состав и команда
 - cast(json)
 - crew(json)
 - id(int)

keywords Ключевые слова к фильмам
 - id(int) - movie id
 - keywords(json)

links_small содержит связи между  TMDB and IMDB ID
 - movieId(int)
 - imdbId(int)
 - tmdbId(int)
        
ratings_small.csv Содержит 100 000 рейтингов от 700 пользователей на 9000 фильмов
 - userId(int)
 - movieId(int)
 - rating(int)
 - timestamp(int)

Преобразования признаков: tfidf

Модель: По похожему контенту и совместная фильтрация

### Клонируем репозиторий и создаем образ
```
$ git clone https://github.com/irmatovpv/GB_docker_flask_example.git
$ cd GB_docker_flask_example
$ 

```

### Запускаем контейнер

Здесь Вам нужно создать каталог локально и сохранить туда предобученную модель (<your_local_path_to_pretrained_models> нужно заменить на полный путь к этому каталогу)
```
$ docker run -d -p 8180:8180 -p 8181:8181 irmatovpv/gb_docker_flask_example
```

### Переходим на localhost:8181
