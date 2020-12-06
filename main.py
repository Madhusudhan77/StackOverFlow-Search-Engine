#!/usr/bin/env pythonfrom pprint 
from pprint import pprint as pp 
from flask import Flask, flash, redirect, render_template, request, url_for 
#from SearchEngineApp import query_api
from SearchResult import SearchEngine
app = Flask(__name__)
@app.route('/')
def index():    
    return render_template('Search.html')

@app.route("/result" , methods=['GET', 'POST'])

def result():    
    data =[]
    error = None   
    select = request.form.get('SearchQuery')
    resp = SearchEngine(select)
    '''pp(resp)    
    if resp:
        data.append(resp) 
        
    if len(data) != 2:
        error = 'Bad Response from Search API'''
            
    return render_template('result.html',data=resp,error=error)

if __name__=='__main__':    app.run(debug=True)




