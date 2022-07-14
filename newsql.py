import os
import psycopg2
from astropy.io import fits as pyfits
import datetime
import psycopg2.extras
import numpy as np


class MyDatabase():
    def __init__(self, db="saguaro", user="saguaro"):
        connstring = open('/home/saguaro/software/webapptesting/sassy.conn').readline()
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
        connstring = open('/home/saguaro/software/webapptesting/sassy.conn').readline()
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


def get_or_create_target(ra, dec, radius=2.):
    db = Dictdb()
    res = db.queryfetchall(f"SELECT * FROM tom_targets_target "
                           f"WHERE q3c_radial_query(ra, dec, {ra:f}, {dec:f}, {radius / 3600.:f});")

    if len(res) == 0:
        res = db.queryfetchall(f"INSERT INTO tom_targets_target (name, type, created, modified, ra, dec, epoch, scheme) "
                               f"VALUES ('unconfirmed candidate', 'SIDEREAL', NOW(), NOW(), {ra:f}, {dec:f}, 2000, '') "
                               f"RETURNING *;")
        targetid = res['id'][0]
        temporary_name = f"Target {targetid:d}"
        db.query(f"UPDATE tom_targets_target SET name='{temporary_name}' where id={targetid:d};")
        res['name'][0] = temporary_name
        db.commit()
        db.close()

    return res


def ingestcandidateswithidreturn(number,filename,elongation,ra,dec,fwhm,snr,mag,magerr,rawfilename,obsdate,field,classification,cx,cy,cz,htm16id,targetid,mjdmid,mlscore,ncomb,match):
    db=Dictdb()
    ingestdate="'"+str(datetime.datetime.now())+"'"
    if match ==-1:ret=db.queryfetchall("INSERT INTO candidates (candidatenumber,filename,elongation,ra,dec,fwhm,snr,mag,magerr,rawfilename,obsdate,field,classification,cx,cy,cz,htm16id,targetid,mjdmid,mlscore,ncombine) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id;" % (number, "'"+filename+"'",elongation,ra,dec,fwhm,snr,mag,magerr,"'"+rawfilename+"'","'"+obsdate+"'","'"+field+"'",classification,cx,cy,cz,htm16id,targetid,mjdmid,mlscore,ncomb))
    if match !=-1:ret=db.queryfetchall("INSERT INTO candidates (candidatenumber,filename,elongation,ra,dec,fwhm,snr,mag,magerr,rawfilename,obsdate,field,classification,cx,cy,cz,htm16id,targetid,mjdmid,mlscore,ncombine,gladeid) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id;" % (number, "'"+filename+"'",elongation,ra,dec,fwhm,snr,mag,magerr,"'"+rawfilename+"'","'"+obsdate+"'","'"+field+"'",classification,cx,cy,cz,htm16id,targetid,mjdmid,mlscore,ncomb,match))
    db.commit()
    db.close()
    return ret

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

