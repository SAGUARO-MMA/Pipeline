import psycopg2
import psycopg2.extras
from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta
import numpy as np
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


def get_or_create_targets(ra, dec, radius=2.):
    db = Dictdb()
    values_to_add = ', '.join([str(coords) for coords in zip(ra, dec)])
    query = f"""
    CREATE TEMP TABLE possible_targets (ra double precision, dec double precision);
    INSERT INTO possible_targets VALUES {values_to_add};
    SELECT s.id FROM possible_targets AS p LEFT JOIN LATERAL (
    SELECT t.id FROM tom_targets_target AS t WHERE q3c_join(p.ra, p.dec, t.ra, t.dec, {radius / 3600.:f})
    ORDER BY q3c_dist(p.ra, p.dec, t.ra, t.dec) ASC LIMIT 1) AS s ON true;
    """
    res = db.queryfetchall(query)
    target_ids = np.array(res['id'])

    no_match = target_ids == np.array(None)
    if no_match.any():
        ra_to_add = ra[no_match]
        dec_to_add = dec[no_match]
        coord = SkyCoord(ra_to_add, dec_to_add, unit='deg')
        names_to_add = 'J' + np.char.array(coord.ra.to_string('hourangle', sep='', precision=2, pad=True)) \
                           + np.char.array(coord.dec.to_string('deg', sep='', precision=1, pad=True, alwayssign=True))
        names_to_add = np.char.replace(names_to_add, '.', '')
        values_to_add = ', '.join([f"('{name}', 'SIDEREAL', NOW(), NOW(), {alpha:f}, {delta:f}, 2000, '')"
                                   for name, alpha, delta in zip(names_to_add, ra_to_add, dec_to_add)])
        query = f"""
        INSERT INTO tom_targets_target (name, type, created, modified, ra, dec, epoch, scheme)
        VALUES {values_to_add} RETURNING id;
        """
        res = db.queryfetchall(query)
        target_ids[no_match] = res['id']

        values_to_add = ', '.join([f"({tid}, 14, 1, 55), ({tid}, 14, 1, 56), ({tid}, 14, 1, 57)" for tid in res['id']])
        query = f"""
        INSERT INTO guardian_groupobjectpermission (object_pk, content_type_id, group_id, permission_id)
        VALUES {values_to_add};
        """
        db.query(query)

    db.commit()
    db.close()
    return target_ids


def ingestcandidates(image_data, obsid, dateobs):
    db = Dictdb()
    values_to_add = ', '.join([
        '({NUMBER}, {ELONGATION}, {ALPHAWIN_J2000}, {DELTAWIN_J2000}, {FWHM_TRANS}, {S2N}, {MAG_PSF}, {MAGERR_PSF}, '
        '{CLASSIFICATION}, {CX}, {CY}, {CZ}, {TARGETID}, {MLSCORE}, {MLSCORE_BOGUS}, {MLSCORE_REAL}, {obsid})'.format(
            **row, obsid=obsid) for row in image_data
    ])
    db.query(f"INSERT INTO candidates (candidatenumber, elongation, ra, dec, fwhm, snr, mag, magerr, classification, "
             f"cx, cy, cz, targetid, mlscore, mlscore_bogus, mlscore_real, observation_record_id) "
             f"VALUES {values_to_add} RETURNING id;")

    # the .item() is needed to convert any np.float32 to np.float64, which is JSON serializable
    values_to_add = ', '.join([f"('photometry', 'SAGUARO pipeline', 'SAGUARO pipeline', '{dateobs.iso}', '"
                               + json.dumps({'magnitude': row['MAG_PSF'].item(), 'error': row['MAGERR_PSF'].item(),
                                             'filter': 'Clear', 'instrument': 'CSS'})
                               + f"', {row['TARGETID']})" for row in image_data])
    db.query(f"INSERT INTO tom_dataproducts_reduceddatum (data_type, source_name, source_location, timestamp, value, "
             f"target_id) VALUES {values_to_add}")
    db.commit()
    db.close()


def add_observation_record(basefile, hdr, log=None):
    prev_midnight_utc = Time(hdr['DATE-OBS'])
    next_midnight_utc = prev_midnight_utc + TimeDelta(1., format='jd')
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
    res = db.queryfetchall(f"SELECT id, parameters, observation_id, status "
                           f"FROM tom_surveys_surveyobservationrecord "
                           f"WHERE survey_field_id = '{field}' "
                           f"AND scheduled_start > '{prev_midnight_utc.iso}' "
                           f"AND scheduled_start < '{next_midnight_utc.iso}'")
    if res and res['status'][0] == 'PENDING':  # requested observation
        parameters.update(res['parameters'][0])
        db.query(f"UPDATE tom_surveys_surveyobservationrecord "
                 f"SET parameters='{json.dumps(parameters)}', observation_id='{basefile}', status='COMPLETED', "
                 f"scheduled_start='{dateobs.iso}', modified=NOW() "
                 f"WHERE id={res['id'][0]}")
        if log is not None:
            log.info(f"Associating with requested SurveyObservationRecord {res['id'][0]}")
    elif not res or res['observation_id'][0] != basefile:  # new serendipitous observation
        res = db.queryfetchall(f"INSERT INTO tom_surveys_surveyobservationrecord (facility, parameters, "
                               f"observation_id, status, scheduled_start, created, modified, survey_field_id) "
                               f"VALUES ('CSS', '{json.dumps(parameters)}', '{basefile}', 'COMPLETED', "
                               f"'{dateobs.iso}', NOW(), NOW(), '{field}') RETURNING id")
        if log is not None:
            log.info(f"Creating new serendipitous SurveyObservationRecord {res['id'][0]}")
    else:
        db.query(f"DELETE FROM candidates WHERE observation_record_id={res['id'][0]};")
        if log is not None:
            log.info(f"Associating with existing SurveyObservationRecord {res['id'][0]}. Clearing old candidates.")
    db.commit()
    db.close()
    return res['id'][0], dateobs
