# Import dependencies
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from textblob import Word
from textblob import TextBlob

# generating all page urls
urls = [f"https://www.yelp.com/biz/starbelly-san-francisco?start={10+x*10}#reviews" for x in range(0, 50)]
 
# function to scrape reviews from each url
def scrape_reviews():
    regex = re.compile("raw__")
    reviews = []
    for url in urls:
        try:
            request = requests.get(url)
            soup = BeautifulSoup(request.text, "html.parser")
            res = soup.find_all("span", {"lang": "en"}, class_=regex)
            reviews = [*reviews, *[r.text for r in res]]
        except:
            print("Error in scraping url", url)
    return reviews

# preprocess collected reviews
def preprocess(reviews):
    df = pd.DataFrame(np.array(reviews), columns=["review"])
    stop_words = stopwords.words("english")

    # lowercase
    df["review lower"] = df["review"].apply(lambda x: " ".join(x.lower() for x in x.split()))
    # strip punctuation
    df["review nopunc"] = df['review lower'].str.replace("[^\w\s]", "")
    # remove stopwords
    df["review nostop"] = df["review nopunc"].apply(lambda x: " ".join(x for x in x.split() if x not in stop_words))
    # custom stopwords
    other_stopwords = ["one", "came", "would", "us", "got", "get", "go", "im", "try"]
    df["review noother"] = df["review nostop"].apply(lambda x: " ".join(x for x in x.split() if x not in other_stopwords))
    # lemmatization
    df["cleaned review"] = df["review noother"].apply(lambda x: " ".join(Word(word).lemmatize() for word in x.split()))

    return df

# calculate sentiment
def calculate_sentiment(df):
    df["polarity"] = df["cleaned review"].apply(lambda x: TextBlob(x).sentiment[0])
    df["subjectivity"] = df["cleaned review"].apply(lambda x: TextBlob(x).sentiment[1])
    return df

if __name__ == "__main__":
    reviews = scrape_reviews()
    df = preprocess(reviews)
    sentiment_df = calculate_sentiment(df)
    sentiment_df.to_csv("data/results.csv", index=False)

