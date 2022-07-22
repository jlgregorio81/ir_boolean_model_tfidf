
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from math import log

#the dataset or collection
dataset = [
    "Earth orbits Sun, and the Moon orbits Earth. The Moon is a natural satellite that reflects the Sun's light.",
    "The Sun is the biggest celestial body in our solar system. The Sun is a star. The Sun is beautiful.",
    "Earth is the third planet in our solar system.",
    "The Sun orbits the Milky Way galaxy!"
]

#a variable to store the stopwords
stopWords = stopwords.words('english')
#the lemmatizer 
lemmatizer = WordNetLemmatizer()

#an array to store the clean dataset, after the preprocessing
cleanDataset = []

#perform the precprocessing
#for each doc present in dataset, do...
for doc in dataset:
    #tokenize (a doc is an array of words) the doc and convert to lower case
    tokenDoc = word_tokenize(doc.lower())
    #a variable to store a clean doc
    cleanDoc = []
    #for each word in tokenDoc, do...
    for word in tokenDoc:
        #if the word is alphanumeric and is not present in stopwords, then...
        if(word.isalnum() and word not in stopWords):
            #extracts the lemma and add it to cleanDoc array
            cleanWord = lemmatizer.lemmatize(word)
            cleanDoc.append(cleanWord)
    #add the processed doc to cleanDataset array
    cleanDataset.append(cleanDoc)

print(cleanDataset)

#a function to calculate term frequency
def termFrequency(term, doc):
    count = 0
    #for each word in doc, do...
    for word in doc:
        #if the term is the same as the word, then increment count
        if term == word:
            count += 1
    return count

# tf = termFrequency("sun", cleanDataset[1])
# print(f"TF: {tf}")

#a function to calculate the inverse term frequency
def inverseTermFrequency(term, dataset):
    freq = 0
    #get the number of elements
    size = len(dataset)
    #for each doc in dataset, do...
    for doc in dataset:
        #if term is present in doc, increment the frequency...
        if term in doc:
            freq += 1
    #return the result - using log at base 10
    return log(size/freq, 10)

# idf = inverseTermFrequency("sun", cleanDataset)
# print(f"IDF: {idf}")

#a function to generate the TF-IDF scores
def tfidfScores(term, dataset):
    #calculate the inverse term frequency
    idf = inverseTermFrequency(term, dataset)
    #an array to store the scores
    scores = []
    #a variable to create the id of the document
    id = 1
    #for each doc in dataset, do...
    for doc in dataset:
        #calculate the tfidf for doc
        tfidf = termFrequency(term, doc) * idf
        #add the score in scores list
        scores.append({"id" : id, "score" : round(tfidf,5)})
        #increment the id
        id += 1 
    return scores

#generate the tfidf scores
tfidf = tfidfScores("sun", cleanDataset)

#create a function to define the key to sort
def mySort(obj):
    return obj['score']

#sort the scores
tfidf.sort(key=mySort, reverse=True)
#print the result
print(tfidf)
