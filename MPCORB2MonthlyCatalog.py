from datetime import datetime

raw_catalog_list = []
f = open("MPCORB.DAT", "r")
for line in f:
    if line == "\n":
        continue  # in case there's a blank
    else:  # line in the original data file
        raw_catalog_list.append(line.rstrip())
f.close()

for n in range(100):
    if raw_catalog_list[n] == '-' * 160:
        start_line = n + 1
        # to define the start of the actual data table,
        # which comes after ~30 lines of header text

cropped_catalog_list = []
# crop off the header
for n in range(len(raw_catalog_list) - start_line):
    cropped_catalog_list.append(raw_catalog_list[n + start_line])

full_catalog = []

for obj_mpc in cropped_catalog_list:
    abs_m_H = obj_mpc[8:14].strip()
    slope_G = obj_mpc[14:20].strip()
    epoch = obj_mpc[20:26].strip()
    mean_anomaly_M = obj_mpc[26:36].strip()
    peri = obj_mpc[37:47].strip()
    node = obj_mpc[48:58].strip()
    inclin = obj_mpc[59:69].strip()
    eccen = obj_mpc[70:80].strip()
    motion_n = obj_mpc[80:92].strip()
    a = obj_mpc[92:104].strip()
    unc_U = obj_mpc[105:107].strip()
    readable_designation = obj_mpc[166:194].strip()

    # MPC format has a "packed" date, allowing the epoch to be stored in
    # fewer digits. However, this must be converted to mm/dd/yyyy format
    # for XEphem.
    epoch_x = f'{int(epoch[3], 36):02d}/{int(epoch[4], 36):02d}.0/{int(epoch[0], 36):02d}{epoch[1:3]}'

    if unc_U == "":
        unc_U = "?"
    expanded_designation = readable_designation + " " + unc_U

    # Write XEphem format orbit to the full_catalog list.
    full_catalog.append(expanded_designation + ",e," + inclin + ","
                        + node + "," + peri + "," + a + "," + motion_n + "," + eccen + "," +
                        mean_anomaly_M + "," + epoch_x + "," + "2000" + ",H " + abs_m_H +
                        "," + slope_G + "\n")

now = datetime.utcnow()
f2 = open(f"{now.year:04d}_{now.month:02d}_ORB.DAT", "w")
for obj in full_catalog:
    f2.write(obj)
f2.close()
