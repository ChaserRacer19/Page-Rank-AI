import os
import random
import re
import sys 

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """

    probabilities = dict()
    allLinks = corpus
    accesibleLinks = corpus[page]
    numOfPages = len(accesibleLinks)

    if len(accesibleLinks) == 0:
        for p in allLinks:
            probabilities[p] = 1/len(allLinks)

    for p in allLinks:
        probabilities[p] = (1-damping_factor)/(len(allLinks))

    for p in accesibleLinks:
        probabilities[p] += (damping_factor/numOfPages)
   
    return probabilities

# print(transition_model("corpus0",'2.html',.85))



def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # start = random.randint(1,len)
    sampleCorpus = dict()
    visits = dict()
    startingLink = random.choice(list(corpus))

    visits[startingLink] = 1
    
    pageProbabilties = transition_model(corpus,startingLink,damping_factor)
    pageItems = list(pageProbabilties.keys())
    pageWeights = list(pageProbabilties.values())

    nextLink = random.choices(pageItems,weights=pageWeights,k=1)[0]

    visits[nextLink] = 1
    for i in range(n-2):
        pageProbabilties = transition_model(corpus,nextLink,damping_factor)
        pageItems = list(pageProbabilties.keys())
        pageWeights = list(pageProbabilties.values())

        nextLink = random.choices(pageItems,weights=pageWeights,k=1)[0]

        if nextLink in visits:
            visits[nextLink] += 1
        else:
            visits[nextLink] = 1
    
    visitsList = list(visits.values())
    for i in visits:
        sampleCorpus[i] = (visitsList[list(visits).index(i)])/n

    return sampleCorpus



# print(sample_pagerank("corpus0",.85,100))


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    N = len(list(corpus))
    d = damping_factor
    pageRanks = dict()

    for i in corpus:
        pageRanks[i] = 1/N

    
    while True:
        newPageRanks = dict()
        for p in corpus:
            newPageRanks[p] = (1 - d)/N 
            sum = 0   
            for i in corpus:
                if len(list(corpus[i])) == 0:
                    sum += (pageRanks[i])/N
                elif p in list(corpus[i]):        
                    sum += (pageRanks[i])/len(list(corpus[i]))
                    
            newPageRanks[p] += d * sum

        maxError = float('-inf')
        for p in corpus:
            if abs(pageRanks[p]-newPageRanks[p]) > maxError:
                maxError = pageRanks[p]-newPageRanks[p]

        if maxError<=.001:
            break
        
        # print(pageRanks)
        for p in corpus:
            pageRanks[p] = newPageRanks[p]

    return(pageRanks)
     




if __name__ == "__main__":
    main()
