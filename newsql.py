import psycopg2
import psycopg2.extras
from astropy.coordinates import SkyCoord
from astropy.time import Time
import json


class Dictdb:
    def __init__(self):
        connstring = open('/dataraid6/sassy/software/webapptesting/sassy.conn').readline()
        self.conn = psycopg2.connect(connstring)
        self.cur = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    def query(self, query):
        self.cur.execute(query)

    def queryfetchall(self, query):
        self.cur.execute(query)
        rows = self.cur.fetchall()
        ans = []
        for row in rows:
            ans.append(dict(row))
        ans1 = {k: [d.get(k) for d in ans] for k in {k for d in ans for k in d}}
        return ans1

    def commit(self):
        self.conn.commit()

    def close(self):
        self.cur.close()
        self.conn.close()


def pipecandmatch(basefile):
    db = Dictdb()
    res = db.queryfetchall(f"SELECT filename, candidatenumber FROM candidates WHERE filename = '{basefile}';")
    db.close()
    if res:
        return res['filename'], res['candidatenumber']
    else:
        return [], []


def get_or_create_target(ra, dec, radius=2.):
    db = Dictdb()
    res = db.queryfetchall(f"SELECT * FROM tom_targets_target "
                           f"WHERE q3c_radial_query(ra, dec, {ra:f}, {dec:f}, {radius / 3600.:f});")

    if len(res) == 0:
        coord = SkyCoord(ra, dec, unit='deg')
        temp_name = f"J{coord.ra.to_string('hourangle', sep='', precision=2, pad=True)}" \
                    f"{coord.dec.to_string('deg', sep='', precision=1, pad=True, alwayssign=True)}".replace('.', '')
        res = db.queryfetchall(f"INSERT INTO tom_targets_target (name, type, created, modified, ra, dec, epoch, scheme)"
                               f" VALUES ('{temp_name}', 'SIDEREAL', NOW(), NOW(), {ra:f}, {dec:f}, 2000, '')"
                               f" RETURNING *;")
        for permission_id in [55, 56, 57]:  # view, change, delete
            db.query(f"INSERT INTO guardian_groupobjectpermission (object_pk, content_type_id, group_id, permission_id)"
                     f"VALUES ({res['id'][0]}, 14, 1, {permission_id})")  # give permissions to the "public" group (1)
        db.commit()
        db.close()

    return res


def ingestcandidates(number, filename, elongation, ra, dec, fwhm, snr, mag, magerr, rawfilename, obsdate, field,
                     classification, cx, cy, cz, htm16id, targetid, mjdmid, mlscore, mlbogus, mlreal, ncomb):
    db = Dictdb()
    db.query(f"INSERT INTO candidates (candidatenumber, filename, elongation, ra, dec, fwhm, snr, mag, magerr, "
             f"rawfilename, obsdate, field, classification, cx, cy, cz, htm16id, targetid, mjdmid, mlscore, mlscore_bogus, mlscore_real, ncombine) "
             f"VALUES ({number}, '{filename}', {elongation}, {ra}, {dec}, {fwhm}, {snr}, {mag}, {magerr}, "
             f"'{rawfilename}', '{obsdate}', '{field}', {classification}, {cx}, {cy}, {cz}, {htm16id}, {targetid}, "
             f"{mjdmid}, {mlscore}, {mlbogus}, {mlreal}, {ncomb}) RETURNING id;")
    # the .item() is needed to convert any np.float32 to np.float64, which is JSON serializable
    reduced_datum_value = {'magnitude': mag.item(), 'error': magerr.item(), 'filter': 'Clear', 'instrument': 'CSS'}
    db.query(f"INSERT INTO tom_dataproducts_reduceddatum (data_type, source_name, source_location, timestamp, value, "
             f"target_id) VALUES ('photometry', 'SAGUARO pipeline', 'SAGUARO pipeline', "
             f"'{Time(mjdmid, format='mjd').iso}', '{json.dumps(reduced_datum_value)}', {targetid:d})")
    db.commit()
    db.close()
