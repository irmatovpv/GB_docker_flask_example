# USAGE
# Start the server:
# 	python run_front_server.py
# Submit a request via Python:
#	python simple_request.py

# import the necessary packages
import pandas as pd
import os
import numpy as np
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from surprise import Reader, Dataset, SVDpp, accuracy, SVD
from surprise.model_selection import train_test_split

#import cloudpickle
import flask
import logging
from logging.handlers import RotatingFileHandler
from time import strftime

# initialize our Flask application and the model
app = flask.Flask(__name__)
model = None

handler = RotatingFileHandler(filename='app.log', maxBytes=100000, backupCount=10)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


class Recomender:
	def __init__(self):
		self.md = pd.read_csv('/app/app/data/movies_metadata.csv')
		credits = pd.read_csv('/app/app/data/credits.csv')
		keywords = pd.read_csv('/app/app/data/keywords.csv')
		keywords['id'] = keywords['id'].astype('int')
		credits['id'] = credits['id'].astype('int')
		self.md = self.md[self.md['id'].str.isnumeric()]
		self.md['id'] = self.md['id'].astype('int')
		self.md = self.md.merge(credits, on='id')
		self.md = self.md.merge(keywords, on='id')

		self.links_small = pd.read_csv('/app/app/data/links_small.csv')
		self.id_map = self.links_small[['movieId', 'tmdbId']]
		self.links_small = self.links_small[self.links_small['tmdbId'].notnull()]['tmdbId'].astype('int')

		self.ratings = pd.read_csv('/app/app/data/ratings_small.csv')
		self.content_indices = None
		self.cosine_sim = None
		self.smd = None
		self.indices_map = None
		self.cf = None

		self.__prepare_recommend_content()
		self.__prepare_hybrid()

	def __prepare_smd(self):
		self.md['year'] = pd.to_datetime(self.md['release_date'], errors='coerce').apply(
			lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
		self.md['genres'] = self.md['genres'].fillna('[]').apply(literal_eval).apply(
			lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
		smd = self.md[self.md['id'].isin(self.links_small)]
		s = smd.apply(lambda x: pd.Series(x['keywords']), axis=1).stack().reset_index(level=1, drop=True)
		s.name = 'keyword'
		s = s.value_counts()

		def get_director(x):
			for i in x:
				if i['job'] == 'Director':
					return i['name']
			return np.nan

		def filter_keywords(x):
			words = []
			for i in x:
				if i in s:
					words.append(i)
			return words

		smd['cast'] = smd['cast'].apply(literal_eval)
		smd['crew'] = smd['crew'].apply(literal_eval)
		smd['cast_size'] = smd['cast'].apply(lambda x: len(x))
		smd['crew_size'] = smd['crew'].apply(lambda x: len(x))
		smd['director'] = smd['crew'].apply(get_director)
		smd['director'] = smd['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
		smd['director'] = smd['director'].apply(lambda x: [x, x, x])
		smd['cast'] = smd['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
		smd['cast'] = smd['cast'].apply(lambda x: x[:3] if len(x) >= 3 else x)
		smd['cast'] = smd['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

		stemmer = SnowballStemmer('english')
		smd['keywords'] = smd['keywords'].apply(literal_eval)
		smd['keywords'] = smd['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
		smd['keywords'] = smd['keywords'].apply(filter_keywords)
		smd['keywords'] = smd['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
		smd['keywords'] = smd['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

		smd['soup'] = smd['keywords'] + smd['cast'] + smd['director'] + smd['genres']
		smd['soup'] = smd['soup'].apply(lambda x: ' '.join(x))

		return smd

	def __prepare_recommend_content(self):
		smd = self.__prepare_smd()

		count = CountVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
		smd = smd.reset_index()
		titles = smd['title']
		indices = pd.Series(smd.index, index=smd['title'])
		self.content_indices = indices

		count = CountVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
		count_matrix = count.fit_transform(smd['soup'])
		cosine_sim = cosine_similarity(count_matrix, count_matrix)
		self.cosine_sim = cosine_sim

		self.smd = smd

	def __prepare_hybrid(self):
		def convert_int(x):
			try:
				return int(x)
			except:
				return np.nan

		self.id_map['tmdbId'] = self.id_map['tmdbId'].apply(convert_int)
		self.id_map.columns = ['movieId', 'id']
		self.id_map = self.id_map.merge(self.smd[['title', 'id']], on='id').set_index('title')
		self.indices_map = self.id_map.set_index('id')

		reader = Reader()
		data = Dataset.load_from_df(self.ratings[['userId', 'movieId', 'rating']], reader)
		trainset, testset = train_test_split(data, test_size=.001)
		self.cf = SVD()
		self.cf.fit(trainset)

	def __similar_movies_by_title(self, title):
		indices = self.content_indices
		cosine_sim = self.cosine_sim

		idx = indices[title]
		sim_scores = list(enumerate(cosine_sim[idx]))
		sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
		sim_scores = sim_scores[1:26]
		movie_indices = [i[0] for i in sim_scores]

		return self.smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year', 'id']]

	def recommend_content(self, title):
		def weighted_rating(x):
			v = x['vote_count']
			R = x['vote_average']
			return (v / (v + m) * R) + (m / (m + v) * C)

		movies = self.__similar_movies_by_title(title)

		vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
		vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')
		C = vote_averages.mean()
		m = vote_counts.quantile(0.60)
		qualified = movies[
			(movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())]
		qualified['vote_count'] = qualified['vote_count'].astype('int')
		qualified['vote_average'] = qualified['vote_average'].astype('int')
		qualified['wr'] = qualified.apply(weighted_rating, axis=1)
		qualified = qualified.sort_values('wr', ascending=False).head(10)
		return qualified

	def recommend(self, title, userId=None):
		if userId is None:
			return self.recommend_content(title)

		movies = self.__similar_movies_by_title(title)

		movies['est'] = movies['id'].apply(lambda x: self.cf.predict(userId, self.indices_map.loc[x]['movieId']).est)
		movies = movies.sort_values('est', ascending=False)
		return movies.head(10)

recomender = Recomender()


@app.route("/", methods=["GET"])
def general():
	return """Welcome to fraudelent prediction process. Please use 'http://<address>/predict' to POST"""

@app.route("/predict", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned from the
	# view
	data = {"success": False}
	dt = strftime("[%Y-%b-%d %H:%M:%S]")
	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == "POST":

		title, user_id = "", None
		request_json = flask.request.get_json()
		if request_json["title"]:
			title = request_json['title']

		if request_json["user_id"]:
			user_id = request_json['user_id']

		logger.info(f'{dt} Data: title={title}, user_id={user_id}')
		try:
			preds = recomender.recommend(title, user_id)
		except AttributeError as e:
			logger.warning(f'{dt} Exception: {str(e)}')
			data['predictions'] = str(e)
			data['success'] = False
			return flask.jsonify(data)

		data["predictions"] = preds.to_dict('records')
		# indicate that the request was a success
		data["success"] = True
	# return the data dictionary as a JSON response
	return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading the model and Flask starting server..."
		"please wait until server has fully started"))
	port = int(os.environ.get('PORT', 8180))
	app.run(host='0.0.0.0', debug=False, port=port)
