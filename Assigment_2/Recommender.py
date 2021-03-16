import pandas as pd

class all_functions:
    def __init__(self):
        pass

    def expand_colum(df, column_name, field='title'):
        """ transform a colum with stacked values in a crosstab colum """
        # remove duplicates
        df2 = df.drop_duplicates(subset=['title'])

        # get rid of those pesky []'s and '
        # full_df2['genres2'] = full_df2.genres.apply(lambda x: x.replace('|', ''))
        # full_df2['genres2'] = full_df2.genres.apply(lambda x: x.replace(']', ''))
        # full_df2['genres2'] = full_df2.genres.apply(lambda x: x.replace("'", ""))

        # cast to list
        if df2[column_name].dtype == "int64":
            df2[column_name] = df2[column_name].tolist()
        else:
            df2[column_name] = df2[column_name].str.split('|').tolist()
        # explode into rows
        df = df2.explode(column_name)

        # let's make a crosstab
        df_column = pd.crosstab(df[field], df[column_name])
        df_column

        # return one df with columns by row (util to average statistics for example)
        # return a crosstab table
        return df, df_column

    def calc_distance(cross_df):
        """ receive a crosstab df and return a distance dataframe"""
        # import necessary stuff
        from sklearn.metrics import jaccard_score
        from scipy.spatial.distance import pdist, squareform

        # calculate the distances
        jaccard_distances = pdist(cross_df.values, metric='jaccard')
        square_jaccard_distances = squareform(jaccard_distances)

        # square_jaccard_distances
        jaccard_similarity_array = 1 - square_jaccard_distances
        # jaccard_similarity_array
        distance_df = pd.DataFrame(jaccard_similarity_array, index=cross_df.index, columns=cross_df.index)
        return distance_df

    def get_similarities(distance_df, col_name, numbers=5):
        """receive a distance df and film_name/user_number, and return a list of "numbers" films"""
        # variable defined by user
        # film_name = input('Type the name of your movie:')
        print('Top {} Similar with {}'.format(numbers, col_name))
        print(distance_df[col_name].sort_values(ascending=False).head(numbers))

    def create_combinations(x):
        """to be used inside another functions as apply method"""
        from itertools import permutations

        # create a function that makes combinations of the movie that a user (x) reviewed

        combinations = pd.DataFrame(list(permutations(x.values, 2)), columns=['item_a', 'item_b'])
        return combinations

    def create_combination_df(df, col_filter='userId_x', col_group='title'):
        # use the create_combinations function
        movie_combinations = df.groupby(col_filter)[col_group].apply(create_combinations)

        movie_combinations = movie_combinations.reset_index(drop=True)
        combi_count = movie_combinations.groupby(['item_a', 'item_b']).size()
        combi_count = combi_count.to_frame(name='size').reset_index()

        # remove duplicates (same movie iteam_a and b)
        combi_count['equal'] = combi_count['item_a'] == combi_count["item_b"]
        combi_count = combi_count[combi_count['equal'] == False]

        combi_count.sort_values('size', ascending=False)
        return combi_count

    def cosine_similatirity(df, col_filter='tag'):

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import linear_kernel
        # create object and define stop words language
        tfidf = TfidfVectorizer(stop_words='english')

        # fill na if exists
        df[col_filter] = df[col_filter].fillna('')

        # transform based on object tfidf
        tfidf_matrix = tfidf.fit_transform(df[col_filter])

        # apply functions
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        indices = pd.Series(df.index, index=df['title']).drop_duplicates()

        # return indices and df of similarities
        return indices, cosine_sim

    ### NEED FIX #####
    def get_recommendations_cos(title, df, indices, cosine_sim):
        # Get the index of the movie that matches the title
        idx = indices[title]

        # Get the pairwsie similarity scores of all movies with that movie
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the 10 most similar movies
        sim_scores = sim_scores[1:11]

        # Get the movie indices
        movie_indices = [i[0] for i in sim_scores]
        print('Your movie tags: \n{}'.format(df["tag"].iloc[df[title]]))
        print('Your movie genre: \n{}'.format(df["genres"].iloc[df[title]]))
        # Return the top 10 most similar movies
        return df.iloc[movie_indices]

    def weight_votes(df, group_col, agg_col1='vote_average', agg_col2='vote_count', number=10):
        # calculate average with count
        # avg_rating_df = df.groupby([group_col])[agg_col].agg(['mean', 'count'])
        # avg_rating_df.rename(columns={'mean': 'vote_average', 'count': 'vote_count'}, inplace=True)
        # avg_rating_df.sort_values('vote_average', ascending=False).head(10)

        # C is the mean vote or rating across the whole dataframe
        C = df['vote_average'].mean()

        # m is the minimum votes or ratings to be listed | using a low value because the dataset is small
        m = df['vote_count'].quantile(0.10)

        # weight function (ranked based on number of votes AND score)
        def weighted_rating(x, m=m, C=C):
            v = x['vote_count']
            R = x['vote_average']
            return (v / (v + m) * R) + (m / (m + v) * C)

        # check the new scores
        avg_rating_df = df.copy().loc[df['vote_count'] >= m]
        avg_rating_df['score'] = avg_rating_df.apply(weighted_rating, axis=1)
        avg_rating_df = avg_rating_df.sort_values('score', ascending=False).head(number)
        return avg_rating_df

    def concat_field(df, col_name):
        """ receive a df and a column name, concat multiple rows on that column, for example all rows with different tags"""
        # need to group columns in one row #example tag

        # concatenate the string
        df[col_name] = df.groupby(['title'])[col_name].transform(lambda x: ' '.join(x))

        # drop duplicate data | not using because i prefer df with all rows, so u can made the statics better

        # smaller_selection_tag.drop(columns=['index', 'counts'], inplace=True)
        # smaller_selection_tag = smaller_selection_tag.drop_duplicates(subset=['tag'])
        # smaller_selection_tag.reset_index(inplace=True)

        # show the dataframe
        return smaller_selection_tag

    def create_date_of_ratings(df, col_name):
        pass


