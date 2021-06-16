
from flask import Flask, redirect, url_for, request, jsonify, render_template
import json 
import requests
import re
import pandas as pd
import numpy as np
import nltk
import pickle
from nltk.stem import PorterStemmer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn import ensemble
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict


app = Flask(__name__)


@app.route('/login',methods = ['POST', 'GET'])
def login():
   #if request.method == 'POST':
    #  user = request.form['nm']
     # polota = request.form['descrit']
   return render_template('WebPage1.html')


#@app.route('/postjson', methods = ['POST'])
def postjson(data):
    data= pd.read_csv("Treino_Test_Validacao.csv")
    data['Title-Description'] = data['Title'] +" "+ data['Description']
    data['Title-Description'][0:717]
    stemmer = PorterStemmer()
    stop_words = stopwords.words('portuguese')
    for index,row in data.iterrows():
            filter_titulo = []
            titulo = row['Title-Description']
            titulo = re.sub(r'[^\w\s]',' ',str(titulo)) # limpar
            titulo = titulo.lower()
            words1 = nltk.word_tokenize(titulo) # tokenize
            words1 = [w for w in words1 if not w in stop_words] #stopwords
            for word in words1: 
                filter_titulo.append(stemmer.stem(word))
                data.loc[index,'Title-Description-token'] = str(filter_titulo)
    data.loc[index,'Title-Description-token'] = str(filter_titulo)

    #Convergir para Data=1 e 4
    dataFalse=data[data['Classification']==1]
    dataFalse=dataFalse.reset_index()
    dataTrue=data[data['Classification']==4]
    dataTrue= dataTrue.reset_index()

    #TF-IDF PARA OS DOIS
    unigram_titulo = TfidfVectorizer(max_features = 100, ngram_range=(1,2))
    unigrams1_titulo = unigram_titulo.fit_transform((dataTrue['Title-Description-token']))
    colun = unigram_titulo.get_feature_names()
    uni = pd.DataFrame(unigrams1_titulo.toarray(),columns=unigram_titulo.get_feature_names())
    uni['Index'] = dataTrue['index']
    unigram_titulo_falso = TfidfVectorizer(max_features = 100,ngram_range=(1,2))
    unigrams1_titulo_falso = unigram_titulo_falso.fit_transform((dataFalse['Title-Description-token']))
    colun = unigram_titulo_falso.get_feature_names()
    uni_falso = pd.DataFrame(unigrams1_titulo_falso.toarray(),columns=unigram_titulo_falso.get_feature_names())
    uni_falso['Classification'] = 1 
    uni['Classification'] = 4 
    uni_falso['Index'] = dataFalse['index']
    k=pd.concat([uni_falso, uni], ignore_index=True)
    k=k.fillna(0)



@app.route('/predictmetadados', methods = ['POST'])
def predictMetadados():  
    #Request ao Json
    data = pd.read_csv("Treino_Test_Validacao.csv")
    request_body = request.data 
    news_json = json.loads(request_body)
    news_title = news_json['Title']
    news_description = news_json['Description']
    news_category= news_json['Category']
    news_source = news_json['Source']
    news_result = news_title + " " + news_description
    
    #Juntar a nova notícia às outras
    dataFrame_result = pd.DataFrame({'Title':[news_title],'Description':[news_description],'Title-Description':[news_result], 'Category':[news_category], 'Source':[news_source]})
    dataFrame_result1=pd.concat([data, dataFrame_result], ignore_index=True)

    #Source Limpeza
    dataFrame_result1['Source']=dataFrame_result1['Source'].replace("Bombeiros24", "Bombeiros24.pt")
    dataFrame_result1['Source']=dataFrame_result1['Source'].replace("TugaPress", "Tuga Press")
    dataFrame_result1['Source']=dataFrame_result1['Source'].replace("Eu gosto-e-tu", "Eu-gosto-e-tu")
    dataFrame_result1['Source']=dataFrame_result1['Source'].replace("Sic de Notícias", "SIC")
    dataFrame_result1['Source']=dataFrame_result1['Source'].replace("Sic Notícias", "SIC")
    dataFrame_result1['Source']=dataFrame_result1['Source'].replace("MagazineLusa", "Magazine Lusa")
    dataFrame_result1['Source']=dataFrame_result1['Source'].replace("Maganize Lusa", "Magazine Lusa")
    dataFrame_result1['Source']=dataFrame_result1['Source'].replace("ZAP AEIOU","ZAPAEIOU")
    dataFrame_result1['Source']=dataFrame_result1['Source'].replace("Correio da Manhã Jornal", "Correio da Manhã")
    dataFrame_result1['Source']=dataFrame_result1['Source'].replace("Publico RSS", "Público")
    dataFrame_result1['Source']=dataFrame_result1['Source'].replace("RTP Notícias","RTP")
    dataFrame_result1['Source']=dataFrame_result1['Source'].replace("RTP de Notícias","RTP")
    dataFrame_result1['Source']=dataFrame_result1['Source'].replace("News in Setubal","News in Setúbal")
    dataFrame_result1['Source']=dataFrame_result1['Source'].replace("TSF Notícias","TSF")
    dataFrame_result1['Source']=dataFrame_result1['Source'].replace("Jornal de Negócios","Jornal De Negócios")
    dataFrame_result1['Source']=dataFrame_result1['Source'].replace("Notícias Ao Minuto","Notícias ao Minuto")

    #Categoria Limpeza
    dataFrame_result1['Category'] = dataFrame_result1['Category'].replace('Africa','África')
    dataFrame_result1['Category'] = dataFrame_result1['Category'].replace('África','Mundo')
    dataFrame_result1['Category'] = dataFrame_result1['Category'].replace('Ciência & Saúde','Ciência e Saúde')    
    dataFrame_result1['Category'] = dataFrame_result1['Category'].replace('Notícia','Notícias')
    dataFrame_result1['Category'] = dataFrame_result1['Category'].replace('Na cidade','Na Cidade')
    dataFrame_result1['Category'] = dataFrame_result1['Category'].replace('Mercados num Minuto','Mercados')
    dataFrame_result1['Category'] = dataFrame_result1['Category'].replace('Tech','Tecnologia')
    dataFrame_result1['Category'] = dataFrame_result1['Category'].replace('História','Histórias')
    dataFrame_result1['Category'] = dataFrame_result1['Category'].replace('Saúde e Bem-estar','Saúde')
    dataFrame_result1['Category'] = dataFrame_result1['Category'].replace('Coronavírus','Covid-19')
    dataFrame_result1['Category'] = dataFrame_result1['Category'].replace('Fama','Celebridades')
    dataFrame_result1['Category'] = dataFrame_result1['Category'].replace('Gente','Celebridades')
    dataFrame_result1['Category'] = dataFrame_result1['Category'].replace('Famosos','Celebridades')
    dataFrame_result1['Category'] = dataFrame_result1['Category'].replace('Pessoas','Celebridades')
    dataFrame_result1['Category'] = dataFrame_result1['Category'].replace('Vizela','Portugal')
    dataFrame_result1['Category'] = dataFrame_result1['Category'].replace('Austrália','Mundo')
    dataFrame_result1['Category'] = dataFrame_result1['Category'].replace('Dinheiro Vivo','Dinheiro')
    dataFrame_result1['Category'] = dataFrame_result1['Category'].replace('Futebol','Modalidades')
    dataFrame_result1['Category'] = dataFrame_result1['Category'].replace('Modalidades','Desporto')
    dataFrame_result1['Category'] = dataFrame_result1['Category'].replace('País','Portugal')
    dataFrame_result1['Category'] = dataFrame_result1['Category'].replace('Nacional','Portugal')
    dataFrame_result1['Category'] = dataFrame_result1['Category'].replace('Inovação','Tecnologia')
    dataFrame_result1['Category'] = dataFrame_result1['Category'].replace('Futuro','Tecnologia')
    dataFrame_result1['Category'] = dataFrame_result1['Category'].replace('Vida','Saúde')
    dataFrame_result1['Category'] = dataFrame_result1['Category'].replace('Ciência e Saúde','Saúde')
    dataFrame_result1['Category'] = dataFrame_result1['Category'].replace('Ministério da Cultura','Cultura')
    dataFrame_result1['Category'] = dataFrame_result1['Category'].replace('Internacional','Mundo')
    dataFrame_result1['Category'] = dataFrame_result1['Category'].replace('Internacional','Mundo ')
    dataFrame_result1['Category'] = dataFrame_result1['Category'].replace('Mundo ','Mundo')
    dataFrame_result1['Category'] = dataFrame_result1['Category'].replace('Bolsa','Dinheiro')
    dataFrame_result1['Category'] = dataFrame_result1['Category'].replace('Mercados','Dinheiro')
    
    #LabelEncoder 
    le = preprocessing.LabelEncoder()
    datanew = le.fit(dataFrame_result1['Category'])
    datanew.classes_
    labelencoder = preprocessing.LabelEncoder()
    dataFrame_result1['Category_new'] = labelencoder.fit_transform(dataFrame_result1['Category'])
    print(dataFrame_result1)

    #Tamanho
    for index in range(dataFrame_result1.shape[0]):
        dataFrame_result1.loc[index,'LenTitulo'] = len(dataFrame_result1['Title'].iloc[index])
        dataFrame_result1.loc[index,'LenDescricao'] = len(dataFrame_result1['Description'].iloc[index])
    print(dataFrame_result1)

    #Dataframe
    dataframe = pd.DataFrame(dataFrame_result1, columns=['LenTitulo','Classification','LenDescricao','Category_new'])
    print(dataframe)
    dataframe1 = pd.get_dummies(dataFrame_result1['Source'])
    print(dataframe1)    

    #Ficheiro com o modelo B
    RFC4 = pickle.load(open("modelB.pkl","rb"))
    
    dataframe_previsao = pd.concat([dataframe, dataframe1], axis=1)
    dataframe_previsao = dataframe_previsao.tail(1)
    dataframe_previsao = dataframe_previsao.fillna(0)
    print('\nPrevisao:',dataframe_previsao)
    prediction_proba_rfc = RFC4.predict_proba(dataframe_previsao)
    print('\nProbabilidade da Previsão:',prediction_proba_rfc)
    prediction_proba_rfc * 100
    pred_prob_Multi_ModeloRFC= pd.DataFrame(prediction_proba_rfc, columns = ['CaraterisPROBB1', 'CaraterisPROBB4'])
    print(pred_prob_Multi_ModeloRFC)
    EXPB1 = pd.DataFrame(pred_prob_Multi_ModeloRFC, columns= ['CaraterisPROBB1'])
    EXPB1.to_csv(r'pred_prob_ModeloB.csv', index = False, header=True)

    print(EXPB1)
    #jsonify(probability=str(float(prediction_proba[0][1])))
   
    return 'Sucesso' #jsonify({ "prediction":  str(prediction_proba[0][1])})

    #return jsonify({'probabilities': prediction_proba.tolist()})

@app.route('/predictConteudo', methods = ['POST'])
def predictConteudo():
    data = pd.read_csv("Treino_Test_Validacao.csv")
    #postjson(data3)
    request_body = request.data 
    news_json = json.loads(request_body)
    news_title = news_json['Title']
    news_description = news_json['Description']
    news_result = news_title + " " + news_description

    #Juntar a nova notícia às outras
    dataFrame_result = pd.DataFrame({'Title':[news_title],'Description':[news_description],'Title-Description':[news_result]})
    dataFrame_result1=pd.concat([data, dataFrame_result], ignore_index=True)

    #Pré-processamento
    stemmer1 = PorterStemmer()
    stop_words1 = stopwords.words('portuguese')
    for index,row in dataFrame_result1.iterrows():
        filter_titulo = []
        titulo = row['Title-Description']
        titulo = re.sub(r'[^\w\s]',' ',str(titulo)) # limpar
        titulo = titulo.lower()
        words1 = nltk.word_tokenize(titulo) # tokenize
        words1 = [w for w in words1 if not w in stop_words1] #stopwords
        for word in words1: 
           filter_titulo.append(stemmer1.stem(word))
           dataFrame_result1.loc[index,'Title-Description-token'] = str(filter_titulo)
    dataFrame_result1.loc[index,'Title-Description-token'] = str(filter_titulo)
    
    #Ficheiro com o modelo TF-IDF
    conteudo_transform = pickle.load(open("tf-idf-conteudo.pkl","rb"))
    conteudo_transform1 = pickle.load(open("tf-idf-conteudoFalso.pkl","rb"))
    
    result_conteudo1 = conteudo_transform.transform(dataFrame_result1['Title-Description-token'])
    result_conteudo2 = conteudo_transform1.transform(dataFrame_result1['Title-Description-token'])
    print("\n-----\n", result_conteudo1)
    print("\n-----\n",result_conteudo2)

    colun2 = conteudo_transform.get_feature_names()
    colun1 = conteudo_transform1.get_feature_names()

    nova = pd.DataFrame(result_conteudo1.toarray(),columns=conteudo_transform.get_feature_names())
    print(nova,"\n-----\n")
    nova = nova.fillna(0)

    nova2 = pd.DataFrame(result_conteudo2.toarray(),columns=conteudo_transform1.get_feature_names())
    print(nova2,"\n-----\n")

    nova_noticia = pd.concat([nova,nova2])#, ignore_index=True)
    print("\n-------NOVA NOTÍCIA-----------\n", nova_noticia)

    #Ficheiro com o modelo A
    mnb= pickle.load(open("modeloA.pkl","rb"))
    dataframe_previsao_modeloA = nova_noticia
    dataframe_previsao_modeloA=dataframe_previsao_modeloA.fillna(0)
    dataframe_previsao_modeloA = dataframe_previsao_modeloA.tail(1)
    print('\nPrevisao:\n',dataframe_previsao_modeloA)
   
    prediction_probaA = mnb.predict_proba(dataframe_previsao_modeloA)
    print('\nProbabilidade da Previsão:',prediction_probaA)

    prediction_probaA * 100
    pred_prob_Multi_ModeloMultinomial= pd.DataFrame(prediction_probaA, columns = ['NLPPROBB1', 'NLPPROBB4'])
    print(pred_prob_Multi_ModeloMultinomial)
    EXPA = pd.DataFrame(pred_prob_Multi_ModeloMultinomial, columns= ['NLPPROBB1'])
    EXPA.to_csv(r'pred_prob_ModeloA.csv', index = False, header=True)

    #Modelo C
    dataA = pd.read_csv("pred_prob_ModeloA.csv")
    print(dataA)
    dataB = pd.read_csv("pred_prob_ModeloB.csv")
    print(dataB)

    modeloC = pd.concat([dataA, dataB], axis=1)
    print(modeloC)

    modeloC_final = modeloC
    svm= pickle.load(open("modelo_finalC.pkl","rb"))
    
    modeloC_total = svm.predict_proba(modeloC_final)

    #modeloC_total= modeloC_total * 100
    pred_prob_Multi_MODELOC_ModeloRFC= pd.DataFrame(modeloC_total, columns = ['Global1', 'Global4'])
    pred_prob_Multi_MODELOC_ModeloRFC['Predicted_Class'] = 0 
    print('\n----------MODELO C------\n', pred_prob_Multi_MODELOC_ModeloRFC)

    for i in range(pred_prob_Multi_MODELOC_ModeloRFC.shape[0]):
     if pred_prob_Multi_MODELOC_ModeloRFC.iloc[i]['Global4']<0.30 : 
           pred_prob_Multi_MODELOC_ModeloRFC.loc[i,'Predicted_Class'] = 1
     elif pred_prob_Multi_MODELOC_ModeloRFC.iloc[i]['Global4']<0.70 and pred_prob_Multi_MODELOC_ModeloRFC.iloc[i]['Global4']>=0.30 : 
        pred_prob_Multi_MODELOC_ModeloRFC.loc[i,'Predicted_Class'] = 3
     elif pred_prob_Multi_MODELOC_ModeloRFC.iloc[i]['Global4']>=0.70 : 
        pred_prob_Multi_MODELOC_ModeloRFC.loc[i,'Predicted_Class'] = 4
     
     print('\n----------MODELO C------\n', pred_prob_Multi_MODELOC_ModeloRFC)
    
    return 'Sucesso'



# Correr o servidor de Flask
if __name__ == '__main__':
   #from werkzeug.serving import run_simple
   app.run(debug = False)
   app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
   

#url="http://127.0.0.1:5000/predictmetadados"
#data=json.dumps({"Title":"Migrações: ONG pedem que normas internacionais sejam cumpridas em Ceuta",
#    "Description":"Cinco organizações não-governamentais (ONG) que trabalham com migrantes que chegam à Espanha pediram hoje 'o cumprimento' das normas internacionais sobre direitos humanos e proteção do interesse superior dos menores que estão em Ceuta.",
#    "Category":"Mundo",
#    "Source":"Notícias ao Minuto"   
#  })
#r=requests.post(url,data)
#print(r.json())