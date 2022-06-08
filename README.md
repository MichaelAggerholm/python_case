## Analyse af GoodReads-Books med python

1. [Introduktion](#introduktion)
2. [Libraries](#libraries)
3. [Landekoder](#landekoder)
4. [Top 10 bedst bedømte bøger](#top_10_bedst_bedømte_bøger)
5. [Data baseret på specifikke forfattere](#data_baseret_på_specifikke_forfattere)
6. [Prediction](#prediction)
   1. [Heatmap](#heatmap)
   2. [Linear-Model RidgeClassifier](#linear-Model_ridgeClassifier)
7. [Sources](#sources)


### Introduktion
"Casen" går ud på at analysere data fra et dataset ved brug af python.<br/>
Datasættet jeg har valgt, hedder "GoodReads Books" og her fra vil jeg ved brug af nogle python libraries finde frem til resultaterne som fremgår i dokumentationen herunder.

### Libraries
Libraries jeg har brugt i casen er :<br/>
<ul>
    <li>sklearn</li>
    <li>matplotlib.pyplot</li>
    <li>seaborn</li>
    <li>pandas</li>
    <li>numpy</li>
</ul>

### Landekoder
Først ligger jeg mærke til at der i datasettet fremgår flere sproglag, de er beskrevet som landekoder.<br/>
Først vil jeg finde frem til hvor mange bøger der egentlig opstår med landekoderne 'eng' og 'en-US', da det er disse bøger som er relevante for mig.

Countplot diagram over antallet af bøger i diverse landekoder :<br/>
![](img/language_codes.png)<br/>

Herefter vil jeg gerne have de specifikke bøger med landekoderne 'eng' og 'en-US' ud i et nyt dataframe som jeg kan benytte fremadrettet :
````python
eng_books_df = raw_df[(raw_df['language_code'] == 'eng') | (raw_df['language_code'] == 'en-US')]
````

### Top 10 bedst bedømte bøger

Herefter vil jeg gerne se top 10 af de bedst bedømte engelske bøger i datasettet.

Barplot diagram over de 10 bedst bedømte engelske bøger i datasettet :<br/>
![](img/top_rated_eng_books.png)

### Data baseret på specifikke forfattere

Herfter ligger jeg mærke til at der er to forfattere som jeg gerne vil se data omkring, og gerne kun fra disse to forfattere.<br/>
Derfor finder jeg alle bøger med engelsk sprog lavet af J.K Rowling og/eller J.R.R Tolkien :<br/>
````python
tolkien_Rowling_rows = eng_books_df["authors"].str.contains("J.R.R. Tolkien|J.K. Rowling")
tolkien_Rowling_df = eng_books_df[tolkien_Rowling_rows]
````
![](img/Tolkien_Rowling_books.png)

### Prediction

#### Heatmap
For at finde collaborations har jeg lavet et heatmap over colonner i datasettet som indeholder relevant information :<br/>
![](img/heatmap.png)<br/>

Dog giver det kun mening at tage udgangspunkt i data som har en collaboration over ca. 50<br/>
Derfor har jeg valgt at tage udgangspunkt i text_review_count -> Ratings_count :<br/>
![](img/heatmap_ratings_reviews.png)<br/>

#### Linear-Model RidgeClassifier
Jeg benytter Linear-Model RidgeClassifier fra Sklearn, pointen ved at benytte denne, er at classifieren konverterer outcome til en værdi mellem -1 og 1.
Derefter håndteres resten som Regression model, som undeersøger sammenhængen mellem ratings_count og vores text_reviews_count :

```python
predict = "ratings_count"

x = np.array(df_with_drops.drop([predict], axis=1))
y = np.array(eng_books_df['ratings_count'])

for _ in range(100):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.7)
    ridge = linear_model.RidgeClassifier()

    # Trains model
    ridge.fit(x_train, y_train)

    acc = ridge.score(x_test, y_test)
    print('Acc {0}'.format(acc))
```

### Sources
En del af de steder jeg har fundet hjælp, inspiration og materiale er linket herunder :<br/><br/>
https://www.kaggle.com/datasets/jealousleopard/goodreadsbooks <br/>
https://www.kaggle.com/code/hoshi7/goodreads-analysis-and-recommending-books <br/>
https://www.kaggle.com/code/listonlt/books-data-visualisation-with-seaborn <br/>
https://www.kaggle.com/code/snanilim/book-recommendation-engine <br/>
https://www.machinelearningplus.com/pandas/how-to-create-pandas-dataframe-python/ <br/>
https://stackoverflow.com/questions/53911663/what-does-sklearn-ridgeclassifier-do
