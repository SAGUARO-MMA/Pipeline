import os
import psycopg2
from astropy.io import fits as pyfits
import datetime
import psycopg2.extras
import numpy as np
import htmCircle

class MyDatabase():
    def __init__(self, db="saguaro", user="saguaro"):
        connstring = open('/home/saguaro/software/webapptesting/sql.conn').readline()
        self.conn = psycopg2.connect(connstring)
        self.cur = self.conn.cursor()

    def query(self, query):
        self.cur.execute(query)

    def fetchone(self):
        rows=self.cur.fetchone()
        return rows

    def queryfetchall(self,query):
        self.cur.execute(query)
        rows = self.cur.fetchall()
        return rows

    def selectmatch(self,tablename,column,match1,match2):
#        print("SELECT %s FROM %s WHERE %s = '%s'; " % (column,tablename.lower(), match1, match2))
        self.cur.execute("SELECT %s FROM %s WHERE %s = '%s'; " % (column,tablename.lower(), match1, match2))
        row = [rows[0] for rows in self.cur.fetchall()]
        return row

    def listall(self,tablename):
        self.cur.execute(" SELECT * FROM %s;" % tablename)
        rows = self.cur.fetchall()
#        print(rows)

    def selectcol(self,tablename,column):
#        print("SELECT %s FROM %s; " % (column,tablename.lower()))
        self.cur.execute("SELECT %s FROM %s; " % (column,tablename.lower()))
        row = [rows[0] for rows in self.cur.fetchall()]
        return row

    def commit(self):
        self.conn.commit()

    def close(self):
        self.cur.close()
        self.conn.close()

class Dictdb():
    def __init__(self, db="saguaro", user="saguaro"):
        connstring = open('/home/saguaro/software/webapptesting/sql.conn').readline()
        self.conn = psycopg2.connect(connstring)
        self.cur = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    def query(self, query):
        self.cur.execute(query)

    def queryfetchall(self,query):
        self.cur.execute(query)
        rows = self.cur.fetchall()
        ans = []
        for row in rows:
            ans.append(dict(row))
#        print (ans)
        ans1={k:[d.get(k) for d in ans] for k in {k for d in ans for k in d}}
#        print (ans1)
        return ans1

    def commit(self):
        self.conn.commit()

    def close(self):
        self.cur.close()
        self.conn.close()


def classtest():
    db = MyDatabase()
    res=db.queryfetchall("SELECT * FROM files;")
    print(res)
    db.close()

def alleventstest():
    db = MyDatabase()
    res=db.queryfetchall("SELECT * FROM allevents;")
    print(res)
    db.close()

def gweventstest():
    db = MyDatabase()
    res=db.queryfetchall("SELECT * FROM gwevents;")
    print(res)
    db.close()

def gweventstable():
    db = Dictdb()
    res=db.queryfetchall("SELECT * FROM gwevents;")
    db.close()
    return res

def allcandidatestable():
    db = Dictdb()
    res=db.queryfetchall("SELECT * FROM candidates;")
    db.close()
    return res

def allcandidatesfields(field):
    db = Dictdb()
    res=db.queryfetchall("SELECT * FROM candidates WHERE field = '%s';" % field);
    db.close()
    return res



def alleventstable():
    db = Dictdb()
    res=db.queryfetchall("SELECT * FROM allevents;")
    db.close()
    return res

def gcnevent(name):
    db = Dictdb()
    res=db.queryfetchall("SELECT * FROM allevents WHERE eventname = '%s';" % name)
    db.close()
    return res

def gwevent(name):
    db = Dictdb()
#    print("SELECT * FROM gwevents WHERE gwname = '%s';" % name)
    res=db.queryfetchall("SELECT * FROM gwevents WHERE gwname = '%s';" % name)
    db.close()
    return res

def newgweventfields(today,yesterday):
    db = Dictdb()
#    print("SELECT * FROM gwevents WHERE gwname = '%s';" % name)
    res=db.queryfetchall("SELECT fieldstriggered FROM gwevents WHERE obsdate ='%s' OR obsdate = '%s';" % (today,yesterday))
    db.close()
    return res


def candidate(name):
    db = Dictdb()
#    print("SELECT * FROM gwevents WHERE gwname = '%s';" % name)
    res=db.queryfetchall("SELECT * FROM candidates WHERE id = '%s';" % name)
    db.close()
    return res

def dictfiles(name):
    db = Dictdb()
#    print("SELECT * FROM gwevents WHERE gwname = '%s';" % name)
    res=db.queryfetchall("SELECT * FROM files WHERE filename = '%s';" % name)
    db.close()
    return res

def getingestedfiles(date1,date2):
    db = Dictdb()
    print("SELECT filename FROM files WHERE (obsdate = '%s' OR obsdate = '%s') AND ingested = TRUE;" % (date1,date2))
    res=db.queryfetchall("SELECT filename FROM files WHERE (obsdate = '%s' OR obsdate = '%s') AND ingested = TRUE;" % (date1,date2))
    db.close()
    return res

def setingestedfiles(filename):
    db = Dictdb()
#    print("SELECT * FROM gwevents WHERE gwname = '%s';" % name)
#    res=db.queryfetchall("SELECT filename FROM files WHERE obsdate = '%s' OR obsdate = '%s' and ingested=1;" % (date1,date2))
    db.query("UPDATE files SET ingested = True WHERE filename = '%s'; " % (filename))
    db.commit()
    db.close()
    return

def pipecandmatch(basefile):
    db=MyDatabase()
    files=db.selectmatch('candidates','filename','filename',basefile)
    names=db.selectmatch('candidates','candidatenumber','filename',basefile)
    db.close()
    return files,names

def countcandidates(obsdate):
    db=MyDatabase()
    db.query("SELECT COUNT(*) FROM candidates WHERE obsdate = '%s';" % (obsdate))
    count = db.fetchone()
    db.close()
    return count

def priorityfiles():
    db = Dictdb()
    res=db.queryfetchall("SELECT field FROM priority_reduction where active = True;")
    db.close()
    return res

def droptargetstable():
    db=MyDatabase()
#    db.query("drop index targetindex")
    db.commit()
    db.query("drop table targets")
    db.commit()
    db.close()


def dropprioritytable():
    db=MyDatabase()
    db.query("drop table priorityreduction")
    db.commit()
    db.close()

def droptemplatetable():
    db=MyDatabase()
    db.query("drop table templates;")
    db.commit()
    db.close()
#    print('Templates table deleted')


def dropcandidates():
    db=MyDatabase()
    db.query("drop table candidates;")
    db.query("DROP SEQUENCE targetid_seq;")
#    db.query("drop index candidateindex;")
    db.commit()
    db.close()
    print('Candidates table deleted')


def dropfiletable():
    db=MyDatabase()
    db.query("drop table files;")
    db.commit()
    db.close()
#    print('Files table deleted')

def dropgweventstable():
    db=MyDatabase()
    db.query("drop table gwevents;")
    db.commit()
    db.close()
    print('gwevents table deleted')

def getcandidates(table,field,ra,dec,arcsec,cx,cy,cz):
    db = Dictdb()
    line = htmCircle.htmCircleRegion(16, ra, dec, arcsec)
    line2=' AND (cx * '+str(cx)+' +  cy * '+str(cy)+' + cz * '+str(cz)+' >= cos('+str(np.radians(arcsec/3600.))+') )'
    line3=" AND (field = '"+str(field)+"')  "
    command='select * from '+str(table)+' '+line+' '+line2+' '+line3
    res=db.queryfetchall(command)
    db.close()
    return res

def getcandidatestemp(table,field,ra,dec,arcsec,cx,cy,cz):
    db = Dictdb()
    db.query("CREATE TEMP TABLE temptargets AS SELECT * from targets WHERE field = '"+str(field)+"'; ")
    line = htmCircle.htmCircleRegion(16, ra, dec, arcsec)
    line2=' AND (cx * '+str(cx)+' +  cy * '+str(cy)+' + cz * '+str(cz)+' >= cos('+str(np.radians(arcsec/3600.))+') )'
    command='select * from '+str('temptargets')+' '+line+' '+line2+' '
    res=db.queryfetchall(command)
    db.close()
    return res

#def getcandidatestwo(table,field,ra,dec,arcsec,cx,cy,cz):
#    db = Dictdb()
#    db.queryfetchall("SELECT * from targets WHERE field = '"+str(field)+"'; ")
#    line = htmCircle.htmCircleRegion(16, ra, dec, arcsec)
#    line2=' AND (cx * '+str(cx)+' +  cy * '+str(cy)+' + cz * '+str(cz)+' >= cos('+str(np.radians(arcsec/3600.))+') )'
#    command='select * from '+str('temptargets')+' '+line+' '+line2+' '
#    res=db.queryfetchall(command)
#    db.close()
#    return res

    
def updatetemplates(field,template,tempgood):
    db=MyDatabase()
    db.query("UPDATE templates SET template = '%s', tempgood = '%s' WHERE field = '%s'; " % (template, tempgood, field))
    db.commit()
    db.query("UPDATE files SET template = '%s', tempgood = '%s' WHERE field = '%s'; " % (template, tempgood, field))
    db.commit()
    db.close()
    print('Templates table updated')
    listall('templates')

def updatefieldstriggered(fieldstriggered,gwname):
    db=MyDatabase()
    db.query("UPDATE gwevents SET fieldstriggered = '%s' WHERE gwname = '%s'; " % (fieldstriggered, gwname))
    db.commit()
    db.close()

def createprioritytable():
    tablename='priorityreduction'
    print(tablename,tableexists(tablename))
    if tableexists(tablename)==False:
        db=MyDatabase()
        db.query("""
        CREATE TABLE %s (
        id serial PRIMARY KEY, field VARCHAR NOT NULL, gwevent varchar
        );""" % tablename)
        db.commit()
        db.close()
    

def createfiletable():
    tablename='files'
    print(tablename,tableexists(tablename))
    if tableexists(tablename)==False:
        db=MyDatabase()
        db.query("""
        CREATE TABLE %s (
        id serial PRIMARY KEY, filename VARCHAR NOT NULL, mpccode varchar,
        object VARCHAR NOT NULL, mjd VARCHAR, mjdmid VARCHAR,
        obsdate VARCHAR, ra VARCHAR, dec VARCHAR,
        seqnum VARCHAR, magzp VARCHAR, vphotoff VARCHAR,
        vphoterr VARCHAR, detect50 VARCHAR,
        wcssep VARCHAR, wcsmatch VARCHAR, ingestdate VARCHAR,
        template VARCHAR, tempgood VARCHAR, update VARCHAR
        );""" % tablename)
        db.commit()
        db.close()
        print('Createfiletable was run.')

def createfieldtable(tablename):
    print(tablename,tableexists(tablename))
    if tableexists(tablename)==False:
        db=MyDatabase()
        db.query("""
        CREATE TABLE %s (
        id serial PRIMARY KEY, filename VARCHAR NOT NULL, mpccode varchar,
        object VARCHAR NOT NULL, mjd VARCHAR, mjdmid VARCHAR,
        obsdate VARCHAR, ra VARCHAR, dec VARCHAR,
        seqnum VARCHAR, magzp VARCHAR, vphotoff VARCHAR,
        vphoterr VARCHAR, detect50 VARCHAR,
        wcssep VARCHAR, wcsmatch VARCHAR, ingestdate VARCHAR,
        template VARCHAR, tempgood VARCHAR, update VARCHAR
        );""" % tablename)
        db.commit()
        db.close()
        print('Field '+tablename+' created.')
    
def createtemplatetable():
    tablename='templates'
    print(tablename,tableexists(tablename))
    if tableexists(tablename)==False:
        db=MyDatabase()
        db.query("""
        CREATE TABLE %s (
        id serial PRIMARY KEY, field VARCHAR NOT NULL, template VARCHAR, tempgood VARCHAR
        );""" % tablename)
        db.commit()
        db.close()
        print('Field '+tablename+' created')

def creategwtable():
    tablename='gwevents'
    print(tablename,tableexists(tablename))
    if tableexists(tablename)==False:
        db=MyDatabase()
        db.query("""
        CREATE TABLE %s (
        id serial PRIMARY KEY, gwname VARCHAR NOT NULL, obstype VARCHAR, alertType VARCHAR, 
        map_png VARCHAR, map_fits VARCHAR, retracted VARCHAR, sites VARCHAR, far VARCHAR, 
        ter VARCHAR, bbh VARCHAR, bns VARCHAR, nsbh VARCHAR, dagroup VARCHAR, distance VARCHAR, fieldstriggered VARCHAR, ingestdate VARCHAR
        );""" % tablename)
        db.commit()
        db.close()
        print('GWevents '+tablename+' created')

def createtargetstable():
    tablename='targets'
    print(tablename,tableexists(tablename))
    if tableexists(tablename)==False:
        db=MyDatabase()
        db.query("""
        CREATE TABLE %s (
        targetid serial PRIMARY KEY, ra float8, dec float8, field VARCHAR, classification VARCHAR, cx float8, cy float8, cz float8, htm16id bigint
        );""" % tablename)
        db.commit()
        db.query("CREATE INDEX targetindex ON targets(field,htm16id);")
        db.commit()  
        db.close()
        print('Targets table created')




def createcandidatetable():
    tablename='candidates'
    print(tablename,tableexists(tablename))
    if tableexists(tablename)==False:
        db=MyDatabase()
#        db.query("CREATE SEQUENCE targetid_seq;")  ## already exists so commented
 #       db.commit()
        db.query("""
        CREATE TABLE %s (
        id serial PRIMARY KEY, candidatenumber VARCHAR NOT NULL, filename VARCHAR, xwin VARCHAR, ywin VARCHAR, 
        errx2win VARCHAR, erry2win VARCHAR, errxywin VARCHAR, elongation VARCHAR, ra VARCHAR, dec VARCHAR, 
        fwhm VARCHAR, snr VARCHAR, fluxpsf VARCHAR, fluxpsferr VARCHAR, mag VARCHAR, magerr VARCHAR, rawfilename VARCHAR, 
        obsdate VARCHAR, field VARCHAR, seqnum VARCHAR, classification VARCHAR, image VARCHAR,reference VARCHAR, diff VARCHAR, 
        scorr VARCHAR, zoomimage VARCHAR, zoomreference VARCHAR, zoomdiff VARCHAR, zoomscorr VARCHAR,ingestdate VARCHAR, 
        cx float8, cy FLOAT8, cz FLOAT8, htm16id BIGINT, targetid BIGINT, datemid VARCHAR, mjdmid FLOAT8
        );""" % tablename)
        db.commit()
        db.query("CREATE INDEX candidateindex ON candidates(field,obsdate);")
        db.commit()
        db.close()
        print('Candidates '+tablename+' created')

def ingestcandidates(number,filename,xwin,ywin,errx2win,erry2win,errxywin,elongation,ra,dec,fwhm,snr,fluxpsf,fluxpsferr,mag,magerr,rawfilename,obsdate,field,seqnum,classification,img,ref,diff,scorr,zoomimg,zoomref,zoomdiff,zoomscorr):
    db=Dictdb()
    ingestdate="'"+str(datetime.datetime.now())+"'"
    res=db.queryfetchall("INSERT INTO candidates (candidatenumber,filename,xwin,ywin,errx2win,erry2win,errxywin,elongation,ra,dec,fwhm,snr,fluxpsf,fluxpsferr,mag,magerr,rawfilename,obsdate,field,seqnum,classification, image, reference, diff, scorr,zoomimage,zoomreference,zoomdiff,zoomscorr,ingestdate) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING *;" % ("'"+number+"'", "'"+filename+"'", "'"+xwin+"'", "'"+ywin+"'", "'"+errx2win+"'", "'"+erry2win+"'", "'"+errxywin+"'", "'"+elongation+"'", "'"+ra+"'", "'"+dec+"'", "'"+fwhm+"'","'"+snr+"'", "'"+fluxpsf+"'", "'"+fluxpsferr+"'", "'"+mag+"'", "'"+magerr+"'","'"+rawfilename+"'","'"+obsdate+"'","'"+field+"'","'"+seqnum+"'", "'"+classification+"'","'"+img+"'","'"+ref+"'","'"+diff+"'", "'"+scorr+"'", "'"+zoomimg+"'","'"+zoomref+"'","'"+zoomdiff+"'", "'"+zoomscorr+"'",ingestdate))
    db.commit()
    db.close()
    return res

def ingestcandidateswithid(number,filename,xwin,ywin,errx2win,erry2win,errxywin,elongation,ra,dec,fwhm,snr,fluxpsf,fluxpsferr,mag,magerr,rawfilename,obsdate,field,seqnum,classification,img,ref,diff,scorr,zoomimg,zoomref,zoomdiff,zoomscorr,cx,cy,cz,htm16id,targetid,datemid,mjdmid,mlscore):
    db=MyDatabase()
    ingestdate="'"+str(datetime.datetime.now())+"'"
    db.query("INSERT INTO candidates (candidatenumber,filename,xwin,ywin,errx2win,erry2win,errxywin,elongation,ra,dec,fwhm,snr,fluxpsf,fluxpsferr, mag, magerr,rawfilename,obsdate,field,seqnum,classification, zoomimage,zoomreference,zoomdiff,zoomscorr,ingestdate, cx,cy,cz,htm16id,targetid,datemid,mjdmid,mlscore) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,%s, %s, %s, %s, %s,%s, %s,%s);" % ("'"+number+"'", "'"+filename+"'", "'"+xwin+"'", "'"+ywin+"'", "'"+errx2win+"'", "'"+erry2win+"'", "'"+errxywin+"'", "'"+elongation+"'", "'"+ra+"'", "'"+dec+"'", "'"+fwhm+"'","'"+snr+"'", "'"+fluxpsf+"'", "'"+fluxpsferr+"'", "'"+mag+"'", "'"+magerr+"'", "'"+rawfilename+"'","'"+obsdate+"'","'"+field+"'","'"+seqnum+"'", "'"+classification+"'","'"+zoomimg+"'","'"+zoomref+"'","'"+zoomdiff+"'", "'"+zoomscorr+"'",ingestdate,cx,cy,cz,htm16id,targetid,"'"+datemid+"'",mjdmid,mlscore))
    db.commit()
    db.close()

def ingestcandidateswithidreturn(number,filename,elongation,ra,dec,fwhm,snr,mag,magerr,rawfilename,obsdate,field,classification,cx,cy,cz,htm16id,targetid,mjdmid,mlscore,ncomb,match):
    db=Dictdb()
    ingestdate="'"+str(datetime.datetime.now())+"'"
    if match ==-1:ret=db.queryfetchall("INSERT INTO candidates (candidatenumber,filename,elongation,ra,dec,fwhm,snr,mag,magerr,rawfilename,obsdate,field,classification,cx,cy,cz,htm16id,targetid,mjdmid,mlscore,ncombine) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id;" % (number, "'"+filename+"'",elongation,ra,dec,fwhm,snr,mag,magerr,"'"+rawfilename+"'","'"+obsdate+"'","'"+field+"'",classification,cx,cy,cz,htm16id,targetid,mjdmid,mlscore,ncomb))
    if match !=-1:ret=db.queryfetchall("INSERT INTO candidates (candidatenumber,filename,elongation,ra,dec,fwhm,snr,mag,magerr,rawfilename,obsdate,field,classification,cx,cy,cz,htm16id,targetid,mjdmid,mlscore,ncombine,gladeid) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id;" % (number, "'"+filename+"'",elongation,ra,dec,fwhm,snr,mag,magerr,"'"+rawfilename+"'","'"+obsdate+"'","'"+field+"'",classification,cx,cy,cz,htm16id,targetid,mjdmid,mlscore,ncomb,match))
    db.commit()
    db.close()
    return ret

#WITH upsert AS (UPDATE updatetest SET value=1 WHERE name='first' RETURNING *) INSERT INTO updatetest (name,value) SELECT 'first', 1 WHERE NOT EXISTS (SELECT * FROM upsert)

def updateingestcandidateswithid(number,filename,xwin,ywin,errx2win,erry2win,errxywin,elongation,ra,dec,fwhm,snr,fluxpsf,fluxpsferr,mag,magerr,rawfilename,obsdate,field,seqnum,classification,img,ref,diff,scorr,zoomimg,zoomref,zoomdiff,zoomscorr,cx,cy,cz,htm16id,targetid,datemid,mjdmid,mlscore):
    db=MyDatabase()
    ingestdate="'"+str(datetime.datetime.now())+"'"
    db.query("WITH upsert AS (UPDATE candidates SET mlscore=%s WHERE candidatenumber=%s AND filename=%s returning *) INSERT INTO candidates (candidatenumber,filename,xwin,ywin,errx2win,erry2win,errxywin,elongation,ra,dec,fwhm,snr,fluxpsf,fluxpsferr, mag, magerr,rawfilename,obsdate,field,seqnum,classification, image, reference, diff, scorr,zoomimage,zoomreference,zoomdiff,zoomscorr,ingestdate, cx,cy,cz,htm16id,targetid,datemid,mjdmid,mlscore) SELECT %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,%s, %s, %s, %s, %s,%s,%s, %s, %s, %s, %s, %s WHERE NOT EXISTS (SELECT * FROM upsert);" % (mlscore,"'"+number+"'", "'"+filename+"'","'"+number+"'", "'"+filename+"'", "'"+xwin+"'", "'"+ywin+"'", "'"+errx2win+"'", "'"+erry2win+"'", "'"+errxywin+"'", "'"+elongation+"'", "'"+ra+"'", "'"+dec+"'", "'"+fwhm+"'","'"+snr+"'", "'"+fluxpsf+"'", "'"+fluxpsferr+"'", "'"+mag+"'", "'"+magerr+"'", "'"+rawfilename+"'","'"+obsdate+"'","'"+field+"'","'"+seqnum+"'", "'"+classification+"'","'"+img+"'","'"+ref+"'","'"+diff+"'", "'"+scorr+"'", "'"+zoomimg+"'","'"+zoomref+"'","'"+zoomdiff+"'", "'"+zoomscorr+"'",ingestdate,cx,cy,cz,htm16id,targetid,"'"+datemid+"'",mjdmid,mlscore))
    db.commit()
    db.close()

def updatemlscore(number,filename,mlscore):
    db=MyDatabase()
    db.query("UPDATE candidates SET mlscore=%s WHERE candidatenumber=%s AND filename=%s;" % (mlscore,"'"+number+"'", "'"+filename+"'"))
    db.commit()
    db.close()


def ingesttargets(ra,dec,field,classification):
    res={}
    cx = np.cos( np.radians(ra) )*np.cos( np.radians(dec))
    cy = np.sin( np.radians(ra) )*np.cos( np.radians(dec))
    cz = np.sin( np.radians(dec) )
    res=getcandidates('targets',field,ra,dec,2,cx,cy,cz)
    if len(res)==0:
        add=True

    if len(res)>=1:
        add=False
        if classification == '1' or classification =='7':res['classification'][0]=classification

    if add:
        # insert in target list
        db=Dictdb()
        cx,cy,cz,htm16id = coordinateident(ra,dec,16)
        ret=db.queryfetchall("INSERT INTO targets (ra,dec,field,classification,cx,cy,cz,htm16id) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING *;" % (ra, dec, "'"+field+"'",classification, cx,cy,cz,htm16id))
        db.commit()
        db.close()
        return ret
    else:
        return res



def ingestgw(gwname, obstype, alertType, map_png, map_fits, retracted, sites, far, ter, bbh, bns, nsbh, group, distance,fieldstriggered):
    db=MyDatabase()
    ingestdate="'"+str(datetime.datetime.now())+"'"
    db.query("INSERT INTO gwevents (gwname, obstype, alertType, map_png, map_fits, retracted, sites, far, ter, bbh, bns, nsbh, dagroup, distance, fieldstriggered, ingestdate) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);" % ("'"+gwname+"'", "'"+obstype+"'", "'"+alertType+"'", "'"+map_png+"'", "'"+map_fits+"'", "'"+retracted+"'", "'"+sites+"'", "'"+far+"'", "'"+ter+"'", "'"+bbh+"'", "'"+bns+"'", "'"+nsbh+"'", "'"+group+"'", "'"+distance+"'", "'"+fieldstriggered+"'", ingestdate))
    db.commit()
    db.close()


def populatetemplate(fits):
    filename=os.path.splitext(fits)[0]
    from astropy.io import fits as pyfits

    hdr = pyfits.getheader(fits,1)
    if hdr.get('OBJECT'):    obj=str(hdr.get('OBJECT'))
    db=MyDatabase()
    if entryexists('templates','field',obj) == False:
        db.query("INSERT INTO templates (field) VALUES ('%s');" % (obj))   
        db.commit()
    else:
         print('Skipping -=- already populated')

    print('Let us see what is in the table.')
    rows=db.queryfetchall(" SELECT * FROM templates;")
    print(rows)
    db.close()

def hastemplate(fits):
    filename=os.path.splitext(fits)[0]
    from astropy.io import fits as pyfits
    hdr = pyfits.getheader(fits,1)
    if hdr.get('OBJECT'):    obj=str(hdr.get('OBJECT'))
    if entryexists('templates','field',obj)==True:
        db=MyDatabase()
        tablename='templates'
        print (tablename)
        print (filename)
        db.query("SELECT COUNT(*) FROM %s WHERE field = '%s' AND WHERE '%s' IS NOT NULL; " % (tablename.lower(), obj,'template'))  ###  <-- this is stupid.  It automatically lowercases it when creating, but not when doing other things??????????

        print("SELECT COUNT(*) FROM %s WHERE filename = '%s'; " % (tablename.lower(), filename))

        rows = db.fetchone()
        db.close()
        if rows[0]==1:
            return True
        else:
            return False
    else:
        return False

def entryexists(tablename,column,entry):
    db=MyDatabase()
    print (tablename,column,entry)
    db.query("SELECT COUNT(*) FROM %s WHERE %s = '%s'; " % (tablename.lower(), column, entry))
    print("SELECT COUNT(*) FROM %s WHERE %s = '%s'; " % (tablename.lower(), column, entry))
    rows = db.fetchone()
    db.close()
    if rows[0]>=1:
        return True
    else:
        return False

def tableexists(tablename):
    db=MyDatabase()
    db.query("""
    SELECT COUNT(*)
    FROM information_schema.tables
    WHERE table_name = '%s';
    """ % tablename.lower() )  ###  <-- this is stupid.  It automatically lowercases it when creating, but not when doing other things??????????

    print("""
    SELECT COUNT(*)
    FROM information_schema.tables
    WHERE table_name = '%s';
    """ % tablename.lower())

    if db.fetchone()[0] == 1:
        db.close()
        print('returning true')
        return True
    print('returning false')
    db.close()
    return False

def insertpriority(field,gwevent):
    db=MyDatabase()
    db.query("INSERT INTO priorityreduction (field, gwevent) VALUES (%s, %s);" % ("'"+field+"'", "'"+gwevent+"'"))
    db.commit()
    db.close()



def insertsqlhdr(tablename,fits,updateentry=False):
    import datetime
    filename=os.path.splitext(fits)[0]
    from astropy.io import fits as pyfits
    hdr = pyfits.getheader(fits,1)
    if hdr.get('GAIN'):    gain="'"+str(hdr.get('GAIN'))+"'"
    else: gain="'NULL'"
    if hdr.get('MPCCODE'):  tel="'"+str(hdr.get('MPCCODE'))+"'"
    else: tel="'NULL'"
    if hdr.get('OBJECT'):  obj="'"+str(hdr.get('OBJECT'))+"'"
    else: obj="'NULL'"
    if hdr.get('MJD'):  mjd="'"+str(hdr.get('MJD'))+"'"
    else: mjd="'NULL'"
    if hdr.get('MJDMID'):  mjdmid="'"+str(hdr.get('MJDMID'))+"'"
    else: mjdmid="'NULL'"
    if hdr.get('DATE-OBS'):  obsdate="'"+str(hdr.get('DATE-OBS'))+"'"
    else: obsdate="'NULL'"
    if hdr.get('CENTRA'):  ra="'"+str(hdr.get('CENTRA'))+"'"
    else: ra="'NULL'"
    if hdr.get('CENTDEC'):  dec="'"+str(hdr.get('CENTDEC'))+"'"
    else: dec="'NULL'"
    if hdr.get('WCSSEP'):  wcssep="'"+str(hdr.get('WCSSEP'))+"'"
    else: wcssep="'NULL'"
    if hdr.get('WCSMATCH'):  wcsmatch="'"+str(hdr.get('WCSMATCH'))+"'"
    else: wcsmatch="NULL"
    if hdr.get('SEQNUM'):  seqnum="'"+str(hdr.get('SEQNUM'))+"'"
    else: seqnum="NULL"
    if hdr.get('MAGZP'):  magzp="'"+str(hdr.get('MAGZP'))+"'"
    else: magzp="NULL"
    if hdr.get('VPHOTOFF'):  vphotoff="'"+str(hdr.get('VPHOTOFF'))+"'"
    else: vphotoff="NULL"
    if hdr.get('VPHOTDEV'):  vphoterr="'"+str(hdr.get('VPHOTDEV'))+"'"
    else: vphoterr="NULL"
    if hdr.get('DETECT50'):  detect50="'"+str(hdr.get('DETECT50'))+"'"
    else: detect50="NULL"

#seqnum VARCHAR
#magzp VARCHAR
#vphotoff VARCHAR
#vphoterr VARCHAR
#detect50, VARCHAR

    template="'NULL'"
    tempgood="'NULL'"
    ingestdate="'"+str(datetime.datetime.now())+"'"
    updated="'NULL'" #str(datetime.datetime.now())
    print('.........')
    print(fits,gain,tel)
    print(filename)
    db=MyDatabase()
#        cur.execute('INSERT INTO %s (origin, destination, duration) VALUES (%s, %s, %s)', (fits, obj, ra))
    if updateentry == False:
        db.query("INSERT INTO files (filename, mpccode, object, mjd, mjdmid, obsdate, ra, dec, seqnum, magzp, vphotoff, vphoterr, detect50, wcssep, wcsmatch, ingestdate, template, tempgood, update) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);" % ('\''+os.path.basename(fits)+'\'',tel,obj,mjd,mjdmid,obsdate,ra,dec,seqnum, magzp, vphotoff, vphoterr, detect50,wcssep,wcsmatch,ingestdate,template,tempgood,updated))   ##########################FIXED TO n09075 for testing
        db.commit()
    if updateentry == True:
        updated="'"+str(datetime.datetime.now())+"'"
        print('TBD')
#        cur.execute("INSERT INTO n09075 (filename, mpccode, object, mjd, mjdmid, obsdate, ra, dec, wcssep, wcsmatch, ingestdate, template, tempgood, update) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON DUPLICATE KEY UPDATE;" % ('\''+os.path.basename(fits)+'\'',tel,obj,mjd,mjdmid,obsdate,ra,dec,wcssep,wcsmatch,ingestdate,template,tempgood,updated))
#        conn.commit()

    print('Let us see what is in the table.')
    rows = db.queryfetchall(" SELECT * FROM %s;" % tablename)
    print(rows)
    db.close



def insertfiletable(fits,updateentry=False):
    import datetime
    filename=os.path.splitext(fits)[0]

    db=MyDatabase()
    dbfiles=db.selectcol('files','filename')
    if (os.path.basename(fits) in dbfiles) == False:

        from astropy.io import fits as pyfits
        hdr = pyfits.getheader(fits,1)
        if hdr.get('MPCCODE'):  tel="'"+str(hdr.get('MPCCODE'))+"'"
        else: tel="'NULL'"
        if hdr.get('OBJECT'):  obj="'"+str(hdr.get('OBJECT'))+"'"
        else: obj="'NULL'"
        if hdr.get('EXPTIME'):  exptime=str(hdr.get('EXPTIME'))
        else: exptime="-1"
        if hdr.get('MJDMID'):  mjdmid=str(hdr.get('MJDMID'))
        else: mjdmid="-1"
        if hdr.get('DATE-OBS'):  obsdate="'"+str(hdr.get('DATE-OBS'))+"'"
        else: obsdate="'1990-01-01'"
        if hdr.get('MAGZP'):  magzp=str(hdr.get('MAGZP'))
        else: magzp="-1"
        if hdr.get('VPHOTOFF'):  vphotoff=str(hdr.get('VPHOTOFF'))
        else: vphotoff="-1"
        if hdr.get('VPHOTDEV'):  vphoterr=str(hdr.get('VPHOTDEV'))
        else: vphoterr="-1"
        if hdr.get('DETECT50'):  detect50=str(hdr.get('DETECT50'))
        else: detect50="-1"
#        cur.execute('INSERT INTO %s (origin, destination, duration) VALUES (%s, %s, %s)', (fits, obj, ra))
        ingestdate="'"+str(datetime.datetime.now())+"'"
        db.query("INSERT INTO files (filename, mpccode, object_name, exptime,mjdmid, obsdate, magzp, vphotoff, vphoterr, detect50,ingestdate) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s,%s,%s);" % ('\''+os.path.basename(fits)+'\'',tel,obj,'30',mjdmid,obsdate,magzp, vphotoff, vphoterr, detect50,ingestdate))
        db.commit()

    db.close()

def insertfiletablearch(fits,updateentry=False):
    import datetime
    filename=os.path.splitext(fits)[0]

    db=MyDatabase()
    dbfiles=db.selectcol('files','filename')
    if (os.path.basename(fits) in dbfiles) == False:

        from astropy.io import fits as pyfits
        hdr = pyfits.getheader(fits,0)
        if hdr.get('MPCCODE'):  tel="'"+str(hdr.get('MPCCODE'))+"'"
        else: tel="'NULL'"
        if hdr.get('OBJECT'):  obj="'"+str(hdr.get('OBJECT'))+"'"
        else: obj="'NULL'"
        if hdr.get('EXPTIME'):  exptime=str(hdr.get('EXPTIME'))
        else: exptime="-1"
        if hdr.get('MJDMID'):  mjdmid=str(hdr.get('MJDMID'))
        else: mjdmid="-1"
        if hdr.get('DATE-OBS'):  obsdate="'"+str(hdr.get('DATE-OBS'))+"'"
        else: obsdate="'1990-01-01'"
        if hdr.get('MAGZP'):  magzp=str(hdr.get('MAGZP'))
        else: magzp="-1"
        if hdr.get('VPHOTOFF'):  vphotoff=str(hdr.get('VPHOTOFF'))
        else: vphotoff="-1"
        if hdr.get('VPHOTDEV'):  vphoterr=str(hdr.get('VPHOTDEV'))
        else: vphoterr="-1"
        if hdr.get('DETECT50'):  detect50=str(hdr.get('DETECT50'))
        else: detect50="-1"
#        cur.execute('INSERT INTO %s (origin, destination, duration) VALUES (%s, %s, %s)', (fits, obj, ra))
        ingestdate="'"+str(datetime.datetime.now())+"'"
        db.query("INSERT INTO files (filename, mpccode, object_name, exptime,mjdmid, obsdate, magzp, vphotoff, vphoterr, detect50,ingestdate) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s,%s,%s);" % ('\''+os.path.basename(fits)+'\'',tel,obj,'30',mjdmid,obsdate,magzp, vphotoff, vphoterr, detect50,ingestdate))
        db.commit()

    db.close()



def coordinateident(ra,dec,lev=16):
#     import htmCircle   # import this module outside to make it faster 
    ident=htmCircle.htmID(lev, float(ra),float(dec))
    cx = np.cos( np.radians(ra) )*np.cos( np.radians(dec))
    cy = np.sin( np.radians(ra) )*np.cos( np.radians(dec))
    cz = np.sin( np.radians(dec) )
    return cx,cy,cz,ident

def getcanfield():
    db = Dictdb()
    command="select * from candidates where field= 'N18076' and obsdate = '2019-02-26';"
    res=db.queryfetchall(command)
    db.close()
    return res

