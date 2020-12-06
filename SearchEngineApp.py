#!/usr/bin/env python
# coding: utf-8
from datetime import datetime
import os
import requests
from flask import Flask, request, jsonify, render_template
import pickle

#from SearchResult import SearchEngine

def query_api(Query):
    Datalist=[1,2,3,4,5]
    Query = request.args.get('Query', None)
    
    try:
        data =requests.get(Query)
            
    except Exception as exc:        
           print(exc) 
              
    
    return Datalist


