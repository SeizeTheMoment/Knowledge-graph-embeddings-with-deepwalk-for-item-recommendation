from SPARQLWrapper import SPARQLWrapper, JSON
import json
sparql = SPARQLWrapper("http://dbpedia.org/sparql")
import time
from tqdm import tqdm
queryRoot = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
"""
queryContent = ["dbo:director","dbo:starring","dbo:distributor",
                "dbo:writer","dbo:musicComposer","dbo:producer",
                "dbo:cinematography","dbo:editing","dct:subject"]
queryParameter = ['?film_director','?film_starring','?film_distributor','?film_writer',
                  '?film_musicComposer','?film_producer','?film_cinematography','?film_editing','?film_subject']
querySentence = []
for i in range(9):
    querySentence.append(queryRoot+"\nSELECT DISTINCT ?film_title ?film_name "+queryParameter[i]+"\nWHERE{\n" \
    "?film_title rdf:type <http://dbpedia.org/ontology/Film> .\n?film_title foaf:name ?film_name .\n?film_title "\
    +queryContent[i] + " " +queryParameter[i] +" .\n?film_title ?name ")
f = open("../ml-1m/movies.dat")
out = open("networkentities.txt",mode='a+',encoding='utf-16')
lines = f.readlines()
ind = 0
for line in tqdm(lines):
    ind+=1
    if ind<3279:
        continue
    alist = line.strip().split("::")
    film_name = alist[1].split(" (")[0]
    for i in range(9):
        q = querySentence[i] + '"'+film_name+'"@en\n}'
        sparql.setQuery(q)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()['results']['bindings']
        for result in results:
            if i == 8:
                res = result[queryParameter[i][1:]]['value'][37:].replace("_"," ")
            else:
                res = result[queryParameter[i][1:]]['value'][28:].replace("_"," ")
            out.write("m"+alist[0]+"::"+res+"\n")




