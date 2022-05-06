import os
import psycopg2
from astropy.io import fits as pyfits
import datetime
import psycopg2.extras
import numpy as np


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
        self.cur.execute("SELECT %s FROM %s WHERE %s = '%s'; " % (column,tablename.lower(), match1, match2))
        row = [rows[0] for rows in self.cur.fetchall()]
        return row

    def listall(self,tablename):
        self.cur.execute(" SELECT * FROM %s;" % tablename)
        rows = self.cur.fetchall()

    def selectcol(self,tablename,column):
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
        ans1={k:[d.get(k) for d in ans] for k in {k for d in ans for k in d}}
        return ans1

    def commit(self):
        self.conn.commit()

    def close(self):
        self.cur.close()
        self.conn.close()



def pipecandmatch(basefile):
    db=MyDatabase()
    files=db.selectmatch('candidates','filename','filename',basefile)
    names=db.selectmatch('candidates','candidatenumber','filename',basefile)
    db.close()
    return files,names

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
        db=Dictdb()
        cx,cy,cz = coordinateident(ra,dec)
        ret=db.queryfetchall("INSERT INTO targets (ra,dec,field,classification,cx,cy,cz) VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING *;" % (ra, dec, "'"+field+"'",classification, cx,cy,cz))
        db.commit()
        db.close()
        return ret
    else:
        return res

def updateingestcandidateswithid(number,filename,xwin,ywin,errx2win,erry2win,errxywin,elongation,ra,dec,fwhm,snr,fluxpsf,fluxpsferr,mag,magerr,rawfilename,obsdate,field,seqnum,classification,img,ref,diff,scorr,zoomimg,zoomref,zoomdiff,zoomscorr,cx,cy,cz,targetid,datemid,mjdmid,mlscore):
    db=MyDatabase()
    ingestdate="'"+str(datetime.datetime.now())+"'"
    db.query("WITH upsert AS (UPDATE candidates SET mlscore=%s WHERE candidatenumber=%s AND filename=%s returning *) INSERT INTO candidates (candidatenumber,filename,xwin,ywin,errx2win,erry2win,errxywin,elongation,ra,dec,fwhm,snr,fluxpsf,fluxpsferr, mag, magerr,rawfilename,obsdate,field,seqnum,classification, image, reference, diff, scorr,zoomimage,zoomreference,zoomdiff,zoomscorr,ingestdate, cx,cy,cz,targetid,datemid,mjdmid,mlscore) SELECT %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,%s, %s, %s, %s, %s,%s,%s, %s, %s, %s, %s, %s WHERE NOT EXISTS (SELECT * FROM upsert);" % (mlscore,"'"+number+"'", "'"+filename+"'","'"+number+"'", "'"+filename+"'", "'"+xwin+"'", "'"+ywin+"'", "'"+errx2win+"'", "'"+erry2win+"'", "'"+errxywin+"'", "'"+elongation+"'", "'"+ra+"'", "'"+dec+"'", "'"+fwhm+"'","'"+snr+"'", "'"+fluxpsf+"'", "'"+fluxpsferr+"'", "'"+mag+"'", "'"+magerr+"'", "'"+rawfilename+"'","'"+obsdate+"'","'"+field+"'","'"+seqnum+"'", "'"+classification+"'","'"+img+"'","'"+ref+"'","'"+diff+"'", "'"+scorr+"'", "'"+zoomimg+"'","'"+zoomref+"'","'"+zoomdiff+"'", "'"+zoomscorr+"'",ingestdate,cx,cy,cz,targetid,"'"+datemid+"'",mjdmid,mlscore))

    db.commit()
    db.close()

def setingestedfiles(filename):
    db = Dictdb()
    db.query("UPDATE files SET ingested = True WHERE filename = '%s'; " % (filename))
    db.commit()
    db.close()
    return

def coordinateident(ra,dec):
    cx = np.cos( np.radians(ra) )*np.cos( np.radians(dec))
    cy = np.sin( np.radians(ra) )*np.cos( np.radians(dec))
    cz = np.sin( np.radians(dec) )
    return cx,cy,cz

def getcandidates(table,field,ra,dec,arcsec,cx,cy,cz):
    db = Dictdb()
    line1=' (cx * '+str(cx)+' +  cy * '+str(cy)+' + cz * '+str(cz)+' >= cos('+str(np.radians(arcsec/3600.))+') )'
    line2=" AND (field = '"+str(field)+"')  "
    command='select * from '+str(table)+' '+line1+' '+line2
    res=db.queryfetchall(command)
    db.close()
    return res

