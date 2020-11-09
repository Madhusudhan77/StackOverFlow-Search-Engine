# Stack OverFlow-Search-Engine
Stack OverFlow Search Engine to get Semantic similar posts for the given Query post using ML and DL techniques

### Problem

Building a search engine based on StackOverflow questions, the search results should include the semantic meaning https://meta.stackexchange.com/questions/138356/how-do-i-download-stack-overflows-data think of scalable architecture and  to reduce the time to return the results.


### Objectives

1) Find top 10 posts which are more similar or sementically similar to the given query posts


### Constraints

1) Low latency

2) Scalability

### Datasets Used

1.WindowsPhone 
2.Sports 
3.Robotics 
4.History 
5.Economics 
6.EarthScience 
7.Chemistry 
8.Biology 
9.Aviation 
10.DBA 


### Approach

1.Data is in XML format for all the Datasets.So converting XML data to tabular format and storing the results in .CSV file

2.Combining all the data from different datasets into single Dataset.Doing the preprocessing of the data to remove stopwords,decontracting the strings,Limmitization and removing other punctuation symbols

3.Taking title and Questions as primary feature ,finding the Cosine Similarity to get all posts which gives semantic similarity to the given Query Post

4.Using Pretrained models UniversalSentenceEncoder,Sbert,ELMO for the word embeddings(Vector representation) for all the Title and Questions data

5.Checking performance of Pretrained Models to get the better results.


### Pretrained Models used in Case Study

1.Universal Sentence Encoder

2.Sbert(Sentence transformers)

3.Embeddings from Language Models (ELMo)


### Experimentation

1.Stacking the Sbert ,ELMO and Universal sentence Encoder models

2.Combining the word embeddings from SBert ,ELMO model,USE and getting the Similar posts which is having semantic similarity with other Posts in the Dataset





