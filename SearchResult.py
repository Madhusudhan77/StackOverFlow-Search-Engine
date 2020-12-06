import pickle
from scipy.sparse import hstack 
#import tqdm.notebook as tq
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from scipy import spatial
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

Dataset =pd.read_csv(r'C:\Users\Reddivari Lalitha\Downloads\OwnCaseStudy\Preprocessed_Balanced_Questions_Dataset.csv')
Dataset_Raw =pd.read_csv(r'C:\Users\Reddivari Lalitha\Downloads\OwnCaseStudy\Questions.csv')

USE_Model = tf.saved_model.load(r"C:\Users\Reddivari Lalitha\Downloads\OwnCaseStudy\USE_Model")
#Elmo_Model = tf.saved_model.load(r"C:\Users\Reddivari Lalitha\Downloads\OwnCaseStudy\Elmo_Model")
Elmo_Model = hub.KerasLayer("https://tfhub.dev/google/elmo/3")



kmeans_Model =open(r"C:\Users\Reddivari Lalitha\Downloads\OwnCaseStudy\Cluster\KMeansModel.pkl","rb")
kmeans =pickle.load(kmeans_Model)
Cluster =open(r"C:\Users\Reddivari Lalitha\Downloads\OwnCaseStudy\Cluster\ClusterDictionary.pkl","rb")
Cluster_Dictionary =pickle.load(Cluster)

#Creating a list to keep all Question,Title and TitleAndQuestion Data for further processing
Question_List =[]
Title_List =[]
TitleAndQuestionList =[]
TitleQuestionListTopic=[]
Topic_List =[]
ID_List =[]
Dataset_Raw_Question_List =[]
Dataset_Raw_Title_List =[]
Dataset_Raw_TitleAndQuestionList =[]
Dataset_Raw_ID_List =[]

#List for predicting results



for ID in range(len(Dataset)):
    try:
        Question_List.append(Dataset['Questions'][ID])
        Title_List.append(Dataset['Title'][ID])
        TitleAndQuestionList.append(Dataset['TitleAndQuestions'][ID])
        Topic_List.append(Dataset['Topic'][ID])
        TitleQuestionListTopic.append(Dataset['TitleQuestionAndTopicCorpus'][ID])
        ID_List.append(ID)
    except KeyError:
        pass  
    
Title_Text_Dictionary =dict(zip(ID_List,Title_List))    
Question_Text_Dictionary=dict(zip(ID_List,Question_List))
TitleAndQuestion_Text_Dictionary=dict(zip(ID_List,TitleAndQuestionList))



for ID in range(len(Dataset_Raw)):
    
    try:
        Dataset_Raw_Question_List.append(Dataset_Raw['Questions'][ID])
        Dataset_Raw_Title_List.append(Dataset_Raw['Title'][ID])
        Dataset_Raw_TitleAndQuestionList.append(Dataset_Raw['TitleAndQuestions'][ID])
        Dataset_Raw_TitleAndQuestionList.append(ID)
    except KeyError:
        pass  

    
#Loading TFIDF Vectorizers

#TFIDF_Title =open(r"C:\Users\Reddivari Lalitha\Downloads\OwnCaseStudy\TFIDF\TFIDF_Title_Fit.pkl","rb")
#TFIDF_Questions =open(r"C:\Users\Reddivari Lalitha\Downloads\OwnCaseStudy\TFIDF\TFIDF_Questions.pkl","rb")
#TFIDF_QuestionAndTitle =open(r"C:\Users\Reddivari Lalitha\Downloads\OwnCaseStudy\TFIDF\TFIDF_TitleAndQuestionList_Vec.pkl","rb")
#TFIDF_Title_Fit=pickle.load(TFIDF_Title)
#TFIDF_Questions =pickle.load(TFIDF_Questions)
#TFIDF_TitleAndQuestionList_Vec=pickle.load(TFIDF_QuestionAndTitle)

TFIDF_vectorizer_Title = TfidfVectorizer(min_df=10,lowercase = False)


TFIDF_vectorizer_Questions = TfidfVectorizer(min_df=10,lowercase = False)

TFIDF_vectorizer_TitleAndQuestions = TfidfVectorizer(min_df=10,lowercase = False)

TFIDF_Title_Fit=TFIDF_vectorizer_Title.fit(Title_List)
TFIDF_Title_Transform=TFIDF_Title_Fit.transform(Title_List)

TFIDF_Questions=TFIDF_vectorizer_Questions.fit(Question_List)
TFIDF_Questions_Transform=TFIDF_Questions.transform(Question_List)

TFIDF_TitleAndQuestionList_Vec =TFIDF_vectorizer_TitleAndQuestions.fit(TitleAndQuestionList)
TFIDF_TitleAndQuestionList_Vec_Transform=TFIDF_TitleAndQuestionList_Vec.transform(TitleAndQuestionList)




#Loading ELMO Vectors
ELMO_Title =open(r"C:\Users\Reddivari Lalitha\Downloads\OwnCaseStudy\ELMO_Model\ELMO_Model_Title_dic_Vec.pkl","rb")
ELMO_Questions =open(r"C:\Users\Reddivari Lalitha\Downloads\OwnCaseStudy\ELMO_Model\ELMO_Model_Question_dic_Vec.pkl","rb")
ELMO_TitleAndQuestions =open(r"C:\Users\Reddivari Lalitha\Downloads\OwnCaseStudy\ELMO_Model\ELMO_Model_QAndT_dic_Vec.pkl","rb")
ELMO_Title_Vectors=pickle.load(ELMO_Title)
ELMO_Question_Vectors =pickle.load(ELMO_Questions)
ELMO_TitleAndQuestion_Vectors=pickle.load(ELMO_TitleAndQuestions)

#Loading USE Vectors
USE_Title =open(r"C:\Users\Reddivari Lalitha\Downloads\OwnCaseStudy\USE_Model\USE_Model_Title_dic_Vec.pkl","rb")
USE_Questions =open(r"C:\Users\Reddivari Lalitha\Downloads\OwnCaseStudy\USE_Model\USE_Model_Question_dic_vec.pkl","rb")
USE_QuestionAndTitle =open(r"C:\Users\Reddivari Lalitha\Downloads\OwnCaseStudy\USE_Model\USE_Model_QuestionsAndTitle_dic_Vec.pkl","rb")
USE_Title_Vectors=pickle.load(USE_Title)
USE_Question_Vectors =pickle.load(USE_Questions)
USE_QuestionAndTitle=pickle.load(USE_QuestionAndTitle)



print("Program is running and Loading of Models and Dictionaries  were complete")

def SemanticSimilarityScore(Sentence,Query_Vector,
                            Title_Vector_Dictionary,Question_Vector_Dictionary,
                            QuestionAndTitle_Vector_Dictionary,
                            ClusterPoints,Title_Text_Dictionary,Question_Text_Dictionary):
    
    SimilarityDict={}
    
    for key in ClusterPoints:
        Vector_Similarity_Title = 1 - spatial.distance.cosine(Query_Vector,Title_Vector_Dictionary[key])
        Vector_Similarity_Question = 1 - spatial.distance.cosine(Query_Vector,Question_Vector_Dictionary[key])
        Vector_Similarity_TitleAndQuestion =1 - spatial.distance.cosine(Query_Vector,QuestionAndTitle_Vector_Dictionary[key])
        
        #Adding Cosine Score from Title,Question and  TitleQuestion
       
        SimilarityDict[key] =Vector_Similarity_Title+Vector_Similarity_Question+Vector_Similarity_TitleAndQuestion
    
    sorted_dic = sorted(SimilarityDict.items(), key=lambda kv: kv[1],reverse=True)[:10]
    
    return SimilarityDict



def ELMO_SemanticScore(Sentence,ClusterPoints):
    
    SentenceArray =np.asarray([Sentence])
    Tensor_Sentence =tf.convert_to_tensor(SentenceArray)
    Query_Vector =np.asarray(Elmo_Model(Tensor_Sentence)).reshape(-1,1)
    ELOM_SimilarityDict=SemanticSimilarityScore(Sentence,Query_Vector,ELMO_Title_Vectors,
                                               ELMO_Question_Vectors,
                                               ELMO_TitleAndQuestion_Vectors,ClusterPoints,Title_Text_Dictionary,Question_Text_Dictionary)
    
    return ELOM_SimilarityDict

def USE_SemanticScore(Sentence,ClusterPoints):
    
    Query_Vector= tf.make_ndarray(tf.make_tensor_proto(USE_Model([Sentence]))).reshape(-1,1)
    Use_SimilarityDict=SemanticSimilarityScore(Sentence,Query_Vector,USE_Title_Vectors,
                                               USE_Question_Vectors,
                                               USE_QuestionAndTitle,ClusterPoints,Title_Text_Dictionary,Question_Text_Dictionary)
    
    return Use_SimilarityDict



def SearchEngine(Sentence):
    
    Predicted_Title=[]
    Predicted_Question=[]
    Sentence=str(Sentence)
    
    Final_SimilarScore_Dict ={}

    Q1=TFIDF_Title_Fit.transform([Sentence])
    Q2=TFIDF_Questions.transform([Sentence])
    Q3=TFIDF_TitleAndQuestionList_Vec.transform([Sentence])
    
    Query_TFIDF_Vec=hstack((Q1,Q2,Q3))
    prediction =kmeans.predict(Query_TFIDF_Vec)

    #Keeping all data points into single dictionary 
    ClusterPoints=Cluster_Dictionary[int(prediction)]
    
    #Calling Respective models and get semantic Scores
    Use_SimilarityDict=USE_SemanticScore(Sentence,ClusterPoints)
    ELOM_SimilarityDict=ELMO_SemanticScore(Sentence,ClusterPoints)

    
    #Combining the similarity scores from all three pretrained models
    for key in Use_SimilarityDict:
        Final_SimilarScore_Dict[key]=Use_SimilarityDict[key]+(ELOM_SimilarityDict[key])
        
        
    #Retriving the top 10 Similar Titles and Questions Posts
    Final_SimilaritySorted_List =  sorted(Final_SimilarScore_Dict.items(), key=lambda kv: kv[1],reverse=True)[:10]  
    
    for i in Final_SimilaritySorted_List:
        Index =i[0]
        #print("SimilarityScore:",i[1])
        #print("TITLE :",Dataset_Raw_Title_List[Index])
        Predicted_Title.append(Dataset_Raw_Title_List[Index])
        Predicted_Question.append(Dataset_Raw_Question_List[Index])
        #print("QUESTION :",Dataset_Raw_Question_List[Index])
        #print("-"*50) 
        
    return Predicted_Title

    



