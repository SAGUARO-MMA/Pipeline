import psycopg2
import psycopg2.extras
from astropy.coordinates import SkyCoord
from astropy.time import Time
import json


class Dictdb:
    def __init__(self):
        self.conn = psycopg2.connect('')  # read from environment variables
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


def pipecandmatch(observation_id):
    db = Dictdb()
    res = db.queryfetchall(f"SELECT candidatenumber FROM candidates WHERE observation_record_id={observation_id};")
    db.close()
    if res:
        return res['candidatenumber']
    else:
        return []


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


def ingestcandidates(number, elongation, ra, dec, fwhm, snr, mag, magerr, classification,
                     cx, cy, cz, targetid, mlscore, mlbogus, mlreal, obsid, dateobs):
    db = Dictdb()
    db.query(f"INSERT INTO candidates (candidatenumber, elongation, ra, dec, fwhm, snr, mag, magerr, classification, "
             f"cx, cy, cz, target_id, mlscore, mlscore_bogus, mlscore_real, observation_record_id) "
             f"VALUES ({number}, {elongation}, {ra}, {dec}, {fwhm}, {snr}, {mag}, {magerr}, {classification}, "
             f"{cx}, {cy}, {cz}, {targetid}, {mlscore}, {mlbogus}, {mlreal}, {obsid}) RETURNING id;")
    # the .item() is needed to convert any np.float32 to np.float64, which is JSON serializable
    reduced_datum_value = {'magnitude': mag.item(), 'error': magerr.item(), 'filter': 'Clear', 'instrument': 'CSS'}
    db.query(f"INSERT INTO tom_dataproducts_reduceddatum (data_type, source_name, source_location, timestamp, value, "
             f"target_id) VALUES ('photometry', 'SAGUARO pipeline', 'SAGUARO pipeline', "
             f"'{dateobs.iso}', '{json.dumps(reduced_datum_value)}', {targetid:d})")
    db.commit()
    db.close()


def add_observation_record(basefile, hdr):
    midnight_utc = Time(hdr['DATE-OBS'])
    dateobs = Time(hdr['DATE-OBS'] + ' ' + hdr['TIME-OBS'])
    field = hdr['OBJECT']
    parameters = {
        'pos_angle': 0.,
        'depth': hdr['T-LMAG'],
        'depth_unit': 'ab_mag',
        'band': 'open',
        'ncombine': hdr['NCOMBINE'],
    }
    db = Dictdb()
    res = db.queryfetchall(f"SELECT id, parameters, status "
                           f"FROM tom_surveys_surveyobservationrecord "
                           f"WHERE survey_field_id = '{field}' "
                           f"AND created > '{midnight_utc.iso}' "
                           f"AND created < '{(midnight_utc + 1).iso}'")
    if res and res['status'][0] == 'PENDING':  # requested observation
        parameters = res['parameters'][0].update(parameters)
        db.query(f"UPDATE tom_surveys_surveyobservationrecord "
                 f"SET parameters='{json.dumps(parameters)}', observation_id='{basefile}', status='COMPLETED', "
                 f"scheduled_start='{dateobs.iso}', modified=NOW() "
                 f"WHERE id={res['id'][0]}")
    elif not res:  # serendipitous observation
        res = db.queryfetchall(f"INSERT INTO tom_surveys_surveyobservationrecord (facility, parameters, "
                               f"observation_id, status, scheduled_start, created, modified, survey_field_id) "
                               f"VALUES ('CSS', '{json.dumps(parameters)}', '{basefile}', 'COMPLETED', "
                               f"'{dateobs.iso}', NOW(), NOW(), '{field}') RETURNING id")
    # otherwise it was already ingested (partially or completely), so just return the ID and dateobs
    return res['id'][0], dateobs
