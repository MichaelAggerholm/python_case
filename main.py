# https://www.kaggle.com/datasets/jealousleopard/goodreadsbooks
# https://www.kaggle.com/code/hoshi7/goodreads-analysis-and-recommending-books
# https://www.kaggle.com/code/listonlt/books-data-visualisation-with-seaborn
# https://www.kaggle.com/code/snanilim/book-recommendation-engine

# https://www.machinelearningplus.com/pandas/how-to-create-pandas-dataframe-python/
# https://stackoverflow.com/questions/53911663/what-does-sklearn-ridgeclassifier-do

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sklearn

from sklearn import linear_model


# Import af bøger som skipper hvis et row har fejl/mangel data :
raw_df = pd.read_csv("books.csv", on_bad_lines='skip')
# print(raw_df)

# Antal bøger i hver landekode :
plt.figure(figsize=(15, 7))
ax = sns.countplot(x=raw_df.language_code, data=raw_df)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x()-0.05, p.get_height()+100))

plt.show()


# Nyt dataframe kun med Engelske bøger :
eng_books_df = raw_df[(raw_df['language_code'] == 'eng') | (raw_df['language_code'] == 'en-US')]
print(eng_books_df)


# Top 10 bedst bedømte bøger i datasettet :
# Nyt dataframe, kun med Engelske bøger som har en antal bedømmelser over én million :
top_eng_books = eng_books_df[eng_books_df['ratings_count'] > 1000000]
# Vælg kun de første 10 bøger, valgt ud fra den bedste gennemsnitlige bedømmmelse :
top_eng_books = top_eng_books.sort_values(by='average_rating', ascending=False).head(10)

sns.set(style="darkgrid")
plt.figure(figsize=(10, 10))

color = sns.color_palette("Set2")
ax = sns.barplot(x="average_rating", y="title", data=top_eng_books, palette=color)

for i in ax.patches:
    ax.text(i.get_width() + .05, i.get_y() + 0.5, str(i.get_width()), fontsize=10, color='k')
plt.show()


# Dataframe med alle Engelske bøger som J.K Rowling og/eller J.R.R Tolkien står som authors på :
tolkien_Rowling_rows = eng_books_df["authors"].str.contains("J.R.R. Tolkien|J.K. Rowling")
tolkien_Rowling_df = eng_books_df[tolkien_Rowling_rows]
print(tolkien_Rowling_df)


# Heatmap
plt.figure(figsize=(15, 10))
df_with_drops = eng_books_df.drop(['bookID', 'title', 'authors', 'average_rating', 'isbn', 'isbn13',
                                   'language_code', '  num_pages', 'publication_date', 'publisher'], axis=1)

sns.heatmap(df_with_drops.corr(), vmin=-1, vmax=1, square=True, annot=True)
plt.show()


# Predict
predict = "ratings_count"

x = np.array(df_with_drops.drop([predict], axis=1))
y = np.array(eng_books_df['ratings_count'])

for _ in range(100):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.5)
    ridge = linear_model.RidgeClassifier()

    # Trains model
    ridge.fit(x_train, y_train)

    acc = ridge.score(x_test, y_test)
    print('Acc {0}'.format(acc))
