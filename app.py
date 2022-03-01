from flask import Flask, render_template, request, make_response, session, redirect, url_for, send_from_directory, json
from functools import wraps
import numpy as np
import os
import sql
import makefc
import deflist
#from config import MEDIA_FOLDER

app=Flask(__name__)

@app.route('/api/<path:filename>')
def download_file(filename):
    return send_from_directory('/home/saguaro/data/', filename, as_attachment=True)

@app.route('/api/finder=<string:name>')
def finder(name):
    db = sql.Dictdb()
    cand=db.queryfetchall("select rawfilename, ra, dec,obsdate  from candidates where targetid = %s order by obsdate desc;" % name)
    db.close()
    os.system('python /home/saguaro/software/BGreduce/finder.py --file %s --ra %s --dec %s --id %s' % (cand['rawfilename'][0], cand['ra'][0], cand['dec'][0], name))

    return send_from_directory('/home/saguaro/data/', 'done.txt', as_attachment=True)

@app.route('/api/remakefinder=<string:name>')
def remakefinder(name):
    db = sql.Dictdb()
    cand=db.queryfetchall("select rawfilename, ra, dec,obsdate  from candidates where targetid = %s order by obsdate desc;" % name)
    db.close()
    os.system('rm /home/saguaro/data/finders/'+name+'_finder.png')
    os.system('python /home/saguaro/software/BGreduce/finder.py --file %s --ra %s --dec %s --id %s' % (cand['rawfilename'][0], cand['ra'][0], cand['dec'][0], name))

    return send_from_directory('/home/saguaro/data/', 'done.txt', as_attachment=True)

@app.route('/api/moplot=<string:name>')
def moplot(name):
    db = sql.Dictdb()
    cand=db.queryfetchall("select rawfilename, ra, dec,obsdate  from candidates where id = %s order by obsdate desc;" % name)
    db.close()
#    print('python /home/saguaro/software/BGreduce/SExt_coords.py --file %s --ra %s --dec %s --id %s' % (cand['rawfilename'][0], cand['ra'][0], cand['dec'][0], name))
    os.system('python3 /home/saguaro/software/BGreduce/SExt_coords.py --file %s --ra %s --dec %s --id %s' % (cand['rawfilename'][0], cand['ra'][0], cand['dec'][0], name))

    return send_from_directory('/home/saguaro/data/', 'done.txt', as_attachment=True)


@app.route('/api/loc=<string:name>')
def loc(name):
#    print('name',name)
    filename=makefc.loc(name)
    return send_from_directory('/home/saguaro/data/', filename, as_attachment=True)

@app.route('/api/proc=<string:name>')
def proc(name):
    name=name.replace(']', '/')
    if deflist.isrunningpy3(name)==True or deflist.isrunningpy2(name)==True:
#        print(name,True)
        return json.dumps(True)
    else:
#        print(name,False)
        return json.dumps(False)
  
@app.route('/api/individ=<string:name>')
def individual(name):
#    print('id',name)
    db = sql.Dictdb()
    cand=db.queryfetchall("select rawfilename, ra, dec  from candidates where id = %s;" % name)
    db.close()
    print('cand',cand)
    print(cand['rawfilename'], cand['ra'], cand['dec'], name)
    print('python /home/saguaro/software/BGreduce/indiv.py --file %s --ra %s --dec %s --id %s' % (cand['rawfilename'][0], cand['ra'][0], cand['dec'][0], name))
    os.system('python /home/saguaro/software/BGreduce/indiv.py --file %s --ra %s --dec %s --id %s' % (cand['rawfilename'][0], cand['ra'][0], cand['dec'][0], name))
  
    return send_from_directory('/home/saguaro/data/', 'done.txt', as_attachment=True)

@app.route('/api/burstindivid=<string:name>')
def burstindividual(name):
#    print('id',name)
    db = sql.Dictdb()
    cand=db.queryfetchall("select rawfilename, ra, dec  from burstcandidates where id = %s;" % name)
    db.close()
    print('cand',cand)
    print(cand['rawfilename'], cand['ra'], cand['dec'], name)
    print('python /home/saguaro/software/BGreduce/indiv.py --file %s --ra %s --dec %s --id %s' % (cand['rawfilename'][0], cand['ra'][0], cand['dec'][0], name))
    os.system('python /home/saguaro/software/BGreduce/indiv.py --file %s --ra %s --dec %s --id %s' % (cand['rawfilename'][0], cand['ra'][0], cand['dec'][0], name))

    return send_from_directory('/home/saguaro/data/', 'done.txt', as_attachment=True)



@app.route('/api/runlimit=<string:name>')
def runlimit(name):
#    print('targetid',name)
    db = sql.Dictdb()
    cand=db.queryfetchall("select field, ra, dec  from candidates where targetid = %s;" % name)
    db.close()    
    print('/////////////////////////////cand',cand)
    if cand=={}:
        db = sql.Dictdb()
        cand=db.queryfetchall("select field, ra, dec  from targets where targetid = %s;" % name)
        db.close()
#    print(cand['rawfilename'], cand['ra'], cand['dec'], name)
#    print('python /home/saguaro/software/BGreduce/thumbnail_history.py --field %s --id %s --ra %s --dec %s' % (cand['field'][0], name, cand['ra'][0], cand['dec'][0]))
    print('python /home/saguaro/software/BGreduce/thumbnail_history_all.py --field %s --id %s --ra %s --dec %s' % (cand['field'][0], name, cand['ra'][0], cand['dec'][0]))
    os.system('python /home/saguaro/software/BGreduce/thumbnail_history_all.py --field %s --id %s --ra %s --dec %s' % (cand['field'][0], name, cand['ra'][0], cand['dec'][0]))
    return send_from_directory('/home/saguaro/data/', 'done.txt', as_attachment=True)


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5013,debug=True)
