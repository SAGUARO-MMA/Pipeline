import psycopg2
import psycopg2.extras


class Dictdb:
    def __init__(self):
        connstring = open('/home/saguaro/software/webapptesting/sassy.conn').readline()
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
        res = db.queryfetchall(f"INSERT INTO tom_targets_target (name, type, created, modified, ra, dec, epoch, scheme)"
                               f" VALUES ('unconfirmed candidate', 'SIDEREAL', NOW(), NOW(), {ra:f}, {dec:f}, 2000, '')"
                               f" RETURNING *;")
        targetid = res['id'][0]
        temporary_name = f"Target {targetid:d}"
        db.query(f"UPDATE tom_targets_target SET name='{temporary_name}' where id={targetid:d};")
        res['name'][0] = temporary_name
        db.commit()
        db.close()

    return res


def ingestcandidates(number, filename, elongation, ra, dec, fwhm, snr, mag, magerr, rawfilename, obsdate, field,
                     classification, cx, cy, cz, htm16id, targetid, mjdmid, mlscore, ncomb):
    db = Dictdb()
    db.query(f"INSERT INTO candidates (candidatenumber, filename, elongation, ra, dec, fwhm, snr, mag, magerr, "
             f"rawfilename, obsdate, field, classification, cx, cy, cz, htm16id, targetid, mjdmid, mlscore, ncombine) "
             f"VALUES ({number}, '{filename}', {elongation}, {ra}, {dec}, {fwhm}, {snr}, {mag}, {magerr}, "
             f"'{rawfilename}', '{obsdate}', '{field}', {classification}, {cx}, {cy}, {cz}, {htm16id}, {targetid}, "
             f"{mjdmid}, {mlscore}, {ncomb}) RETURNING id;")
    db.commit()
    db.close()
