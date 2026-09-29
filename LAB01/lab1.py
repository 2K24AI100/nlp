import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
corpus=[
    "The product performance is amazing and fast",
    "The service was fast and performance was great",
    "Terrible customer service and bad performance"
]
vectorizer=CountVectorizer(stop_words='english')
X=vectorizer.fit_transform(corpus)
vocabulary=vectorizer.get_feature_names_out()
print("Vocabulary:")
print(vocabulary)
df=pd.DataFrame(
    X.toarray(),
    columns=(vocabulary)
)
print("\nBag of words Matrix:")
print(df)