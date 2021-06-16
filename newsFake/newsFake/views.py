# coding=utf-8
"""
Routes and views for the flask application.
"""
from newsFake import app
import pandas as pd 
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

def get_request_body(request):
   body = None
   if request.form:
        body = request.form
        try:
            body = body.to_dict(flat=False)
            
        except:
          pass
   elif request.json:
        body = request.json

   elif request.data:
        try:
            body = json.loads(request.data.strip())

        except:
            pass

   return body

@app.route('/')
def index():
  return render_template('inicio.html',title='Detetor de Notícias Falsas')

@app.route('/',methods=['GET'])
def lista():
    data = pd.read_csv("Treino_Test_Validacao.csv")
    valor =  pd.DataFrame.from_dict(request_body, orient='index').T
    valor = valor.rename(columns = {'Source[]': 'Source', 'Category[]': 'Category'}, inplace = False)
    return render_template('listasNoticias.html',result=valor)
    

@app.route('/verifica')
def verifica():
  return render_template('detetor.html',title='Detetor de Notícias Falsas')
  
def predictMetadados(request_body):  
    #Request ao Json
    data = pd.read_csv("Treino_Test_Validacao.csv")
    print('----BODY------', request_body)
    
    valor =  pd.DataFrame.from_dict(request_body, orient='index').T
    print(valor)
    valor = valor.rename(columns = {'Source[]': 'Source', 'Category[]': 'Category'}, inplace = False)
    print(valor)
    
    #Juntar a nova notícia às outras
    #dataFrame_result = pd.DataFrame({'Title':[news_title],'Description':[news_description], 'Category':[news_category], 'Source':[news_source]})
    dataFrame_result1=pd.concat([data, valor], ignore_index=True)

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
    return 'Sucesso'

#@app.route('/predictConteudo', methods = ['GET','POST'])
def predictConteudo(request_body):
    data = pd.read_csv("Treino_Test_Validacao.csv")
    #postjson(data3)
    #request_body = get_request_body(request)
    print('----BODY------', request_body)
    #print('----BODY2 ------', request.json)
    #print('----BODY3 ------', dict(request.json))
    #print('----BODY4------', format(request.get_json(force=True)))
    #news_json = json.loads(request_body)
    news_title = request_body['Title']
    news_description = request_body['Description']
    news_result = str(news_title) + " " + str(news_description)

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

    return 'Sucesso'

@app.route('/del', methods=['Delete'])
def delet():
    request_body = get_request_body(request)
    del request_body
    

@app.route('/modelofinal', methods = ['GET','POST'])
def modelofinal():
  
        request_body = get_request_body(request)
        news_title = request_body['Title']
        news_description = request_body['Description']
        valor =  pd.DataFrame.from_dict(request_body, orient='index').T
        print(valor)
        valor = valor.rename(columns = {'Source[]': 'Source', 'Category[]': 'Category'}, inplace = False)
        print(valor)

        modeloA=  predictConteudo(request_body)
        modeloB = predictMetadados(request_body)
        dataA = pd.read_csv("pred_prob_ModeloA.csv")
        print(dataA)
        dataB = pd.read_csv("pred_prob_ModeloB.csv")
        print(dataB)

        modeloC = pd.concat([dataA, dataB], axis=1)
        print(modeloC)

        modeloC_final = modeloC
        svm= pickle.load(open("modelo_finalC.pkl","rb"))
    
        modeloC_total = svm.predict_proba(modeloC_final)

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
        
         p = pred_prob_Multi_MODELOC_ModeloRFC[["Predicted_Class"]]
   
         rvalue = pred_prob_Multi_MODELOC_ModeloRFC.Predicted_Class.item()
         
         result=[]
        if rvalue == 1:
           result = ["A notícia pode ser Falsa!", "red"] 
        elif rvalue == 3:
           result = ["A notícia pode ser Duvidosa!!", "brown"]
        elif rvalue == 4:
           result = ["A notícia pode ser Verdadeira!!", "green"]  
 
        return render_template('result.html', result=result)    
 

