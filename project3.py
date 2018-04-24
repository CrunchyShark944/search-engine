import os
import signal
import nltk
from collections import defaultdict
import json
from bs4 import BeautifulSoup
import math
from pymongo import MongoClient


def signal_handler(signum, frame):
    raise Exception("Timeout!")



def getWords():
    words = []
    for i in os.listdir("WEBPAGES_RAW"):
        dirAddr = "WEBPAGES_RAW/" + str(i)
        if os.path.isdir(dirAddr):
            for j in os.listdir(dirAddr):
                fileAddr = "WEBPAGES_RAW/" + str(i) + "/" + str(j)
                print(fileAddr)
                # time limit per file: 2 seconds
                signal.signal(signal.SIGALRM, signal_handler)
                signal.alarm(2)
                try:
                    with open(fileAddr, encoding = "utf-8", mode = "r") as f:
                        words += nltk.word_tokenize(f.read().lower())
                except:
                    continue
                signal.alarm(0)
    return words


def buildGrams(words):
    nGrams = defaultdict(list)
    finder = nltk.collocations.BigramCollocationFinder.from_words(words)
    finder.apply_freq_filter(15)
    finder.apply_word_filter(lambda x: x in nltk.corpus.stopwords.words("english"))
    for i in finder.nbest(nltk.collocations.BigramAssocMeasures().pmi, 30000):
        nGrams[i[0]].append(i[1])
    with open("nGrams.json", mode = "w") as f:
        json.dump(nGrams, f)
    return


def parseFile(nGramsDict, fileAddr, wordFreqCount, totalWordCount):
    with open(fileAddr, encoding = "utf-8", mode = "r") as f:
        text = f.read().lower()
        soup = BeautifulSoup(text, "lxml")
        sentences = nltk.sent_tokenize(text)
        for sentence in sentences:
            words = [x for x in nltk.word_tokenize(sentence) if x not in nltk.corpus.stopwords.words("english")]
            for i in range(len(words)):
                word = words[i]
                # for tf calc
                wordFreqCount[word][fileAddr[13:]] += 1
                totalWordCount[fileAddr[13:]] += 1

                # for 2-grams
                if i + 1 < len(words) and words[i] in nGramsDict and words[i+1] in nGramsDict:
                    word = words[i] + " " + words[i+1]
                    wordFreqCount[word][fileAddr[13:]] += 1
                    totalWordCount[fileAddr[13:]] += 1
    return


def writeToDatabase(collection, wordFreqCount, totalWordCount, totalDocs):
    for word in wordFreqCount.keys():
        info = []
        idf = math.log(totalDocs / len(wordFreqCount[word]))
        for f in wordFreqCount[word].keys():
            tf = wordFreqCount[word][f] / totalWordCount[f]
            print(word, f)
            info.append({"file": f, "count": wordFreqCount[word][f], "tf-idf": tf*idf})
        collection.insert({"word": word, "info": info})
    return


# time limit per file: 5 seconds
def buildDatabase(collection):
    collection.remove()

    nGramsFile = open("nGrams.json", mode = "r")
    nGramsDict = json.load(nGramsFile)
    nGramsFile.close()

    numOfDocs = 0
    wordFreqCount = defaultdict(lambda: defaultdict(int)) # {token: {docid: wordCount}}
    totalWordCount = defaultdict(int) # {docid: totalCount}

    for i in os.listdir("WEBPAGES_RAW"):
        dirAddr = "WEBPAGES_RAW/" + str(i)
        if os.path.isdir(dirAddr):
            for j in os.listdir(dirAddr):
                fileAddr = "WEBPAGES_RAW/" + str(i) + "/" + str(j)
                print(fileAddr)

                signal.signal(signal.SIGALRM, signal_handler)
                signal.alarm(5)
                try:
                    parseFile(nGramsDict, fileAddr, wordFreqCount, totalWordCount)
                except:
                    continue
                signal.alarm(0)
    writeToDatabase(collection, wordFreqCount, totalWordCount, numOfDocs)
    return

if __name__ == "__main__":
    # build n-grams
    boolGram = int(input("Do u wanna build n-grams? Yes: 1, No: 0\n"))
    if boolGram == 10000: # disable feature for demo
        words = getWords()
        buildGrams(words)

    # connect MongoDB
    client = MongoClient("localhost", 27017)
    database = client.searchEngine
    collection = database.words

    # build database
    boolData = int(input("Do u wanna build database? Yes: 1, No: 0\n"))
    if boolData == 10000: # disable feature for demo
        buildDatabase(collection)

    # search search
    bookFile = open("WEBPAGES_RAW/bookkeeping.json", mode = "r")
    bookDict = json.load(bookFile)
    bookFile.close()
    print("**********search engine**********")
    while True:
        query = input("what do u wanna search: (control + c to quit)")
        keys = query.split()
        result = []
        print("*****searching*****")
        if len(keys) == 1:
            for i in collection.find({"word": query}):
                result += [(bookDict[j["file"]], j["tf-idf"]) for j in i["info"]]
            print("*****complete*****")
            for i in sorted(result, key = lambda x: -x[1])[:20]:
                print("URL: {}\ntf-idf: {}\n".format(i[0],i[1]))
        else:
            for i in collection.find({"word": query}):
                result += [(bookDict[j["file"]], j["tf-idf"]*1.5) for j in i["info"]]
            for key in keys:
                for i in collection.find({"word": key}):
                    result += [(bookDict[j["file"]], j["tf-idf"]) for j in i["info"]]
            print("*****complete*****")
            for i in sorted(result, key = lambda x: -x[1])[:20]:
                print("URL: {}\ntf-idf: {}\n".format(i[0],i[1]))
