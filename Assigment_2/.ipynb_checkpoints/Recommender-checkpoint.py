import pandas as pd
# load the dataset small
movies_df = pd.read_csv('ml-small/movies.csv')
ratings_df = pd.read_csv('ml-small/ratings.csv')
links_df = pd.read_csv('ml-small/links.csv')
tags_df = pd.read_csv('ml-small/tags.csv', low_memory=False)

