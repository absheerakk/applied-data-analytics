from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# Sample set of documents

documents = [
    "Data science is an interdisciplinary field.",
    "Machine learning is a part of data science.",
    "Data scientists analyze and interpret complex data.",
    "Python is widely used in machine learning."
]

# Initialize CountVectorizer (BoW model)
vectorizer = CountVectorizer()

# Transform the documents into vectors
X = vectorizer.fit_transform(documents)

# Convert the result into a DataFrame for readability
df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

# Display the document-term matrix
print("Document-Term Matrix (Bag of Words Representation):\n")
print(df)
