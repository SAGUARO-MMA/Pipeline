import sys
import ephem


current_date = ephem.now()
current_year = str(current_date)[:4]
current_month = str(current_date)[5:7]
if current_month == "1/": current_month = "01"
if current_month == "2/": current_month = "02"
if current_month == "3/": current_month = "03"
if current_month == "4/": current_month = "04"
if current_month == "5/": current_month = "05"
if current_month == "6/": current_month = "06"
if current_month == "7/": current_month = "07"
if current_month == "8/": current_month = "08"
if current_month == "9/": current_month = "09"

raw_catalog_list = []
f = open("MPCORB.DAT", "r")
for line in f:
    if line == "\n":
        continue       # in case there's a blank 
    else:              # line in the original data file
        raw_catalog_list.append(line.rstrip())
f.close()

for n in range(100):
    if raw_catalog_list[n] == "----------------------------------------------------------------------------------------------------------------------------------------------------------------":
        start_line = n + 1 
                   # to define the start of the actual data table, 
                   # which comes after ~30 lines of header text

cropped_catalog_list = []
# crop off the header
for n in range(len(raw_catalog_list) - start_line):
    cropped_catalog_list.append(raw_catalog_list[n + start_line])

full_catalog = []

for obj_mpc in cropped_catalog_list:
    # designation = obj_mpc[0:8].strip()
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
    # reference = obj_mpc[107:117].strip()
    # num_obs = obj_mpc[117:123].strip()
    # num_ops = obj_mpc[123:127].strip()
    # opposition_observ_span = obj_mpc[127:137].strip()
    # rms = obj_mpc[137:142].strip()
    # coarse_pert = obj_mpc[142:146].strip()
    # precise_pert = obj_mpc[146:150].strip()
    # computer = obj_mpc[150:161].strip()
    readable_designation = obj_mpc[166:194].strip()
    # date_last_obs = obj_mpc[194:203].strip()
    """
    MPC format has a "packed" date, allowing the epoch to be stored in 
    fewer digits. However, this must be converted to mm/dd/yyyy format 
    for XEphem.
    """
    # month
    if epoch[3:4] == "1": epoch_x = "01/"
    elif epoch[3:4] == "2": epoch_x = "02/"
    elif epoch[3:4] == "3": epoch_x = "03/"
    elif epoch[3:4] == "4": epoch_x = "04/"
    elif epoch[3:4] == "5": epoch_x = "05/"
    elif epoch[3:4] == "6": epoch_x = "06/"
    elif epoch[3:4] == "7": epoch_x = "07/"
    elif epoch[3:4] == "8": epoch_x = "08/"
    elif epoch[3:4] == "9": epoch_x = "09/"
    elif epoch[3:4] == "A": epoch_x = "10/"
    elif epoch[3:4] == "B": epoch_x = "11/"
    elif epoch[3:4] == "C": epoch_x = "12/"

    # day
    if epoch[4:5] == "1": epoch_x = epoch_x + "01.0/"
    elif epoch[4:5] == "2": epoch_x = epoch_x + "02.0/"
    elif epoch[4:5] == "3": epoch_x = epoch_x + "03.0/"
    elif epoch[4:5] == "4": epoch_x = epoch_x + "04.0/"
    elif epoch[4:5] == "5": epoch_x = epoch_x + "05.0/"
    elif epoch[4:5] == "6": epoch_x = epoch_x + "06.0/"
    elif epoch[4:5] == "7": epoch_x = epoch_x + "07.0/"
    elif epoch[4:5] == "8": epoch_x = epoch_x + "08.0/"
    elif epoch[4:5] == "9": epoch_x = epoch_x + "09.0/"
    elif epoch[4:5] == "A": epoch_x = epoch_x + "10.0/"
    elif epoch[4:5] == "B": epoch_x = epoch_x + "11.0/"
    elif epoch[4:5] == "C": epoch_x = epoch_x + "12.0/"
    elif epoch[4:5] == "D": epoch_x = epoch_x + "13.0/"
    elif epoch[4:5] == "E": epoch_x = epoch_x + "14.0/"
    elif epoch[4:5] == "F": epoch_x = epoch_x + "15.0/"
    elif epoch[4:5] == "G": epoch_x = epoch_x + "16.0/"
    elif epoch[4:5] == "H": epoch_x = epoch_x + "17.0/"
    elif epoch[4:5] == "I": epoch_x = epoch_x + "18.0/"
    elif epoch[4:5] == "J": epoch_x = epoch_x + "19.0/"
    elif epoch[4:5] == "K": epoch_x = epoch_x + "20.0/"
    elif epoch[4:5] == "L": epoch_x = epoch_x + "21.0/"
    elif epoch[4:5] == "M": epoch_x = epoch_x + "22.0/"
    elif epoch[4:5] == "N": epoch_x = epoch_x + "23.0/"
    elif epoch[4:5] == "O": epoch_x = epoch_x + "24.0/"
    elif epoch[4:5] == "P": epoch_x = epoch_x + "25.0/"
    elif epoch[4:5] == "Q": epoch_x = epoch_x + "26.0/"
    elif epoch[4:5] == "R": epoch_x = epoch_x + "27.0/"
    elif epoch[4:5] == "S": epoch_x = epoch_x + "28.0/"
    elif epoch[4:5] == "T": epoch_x = epoch_x + "29.0/"
    elif epoch[4:5] == "U": epoch_x = epoch_x + "30.0/"
    elif epoch[4:5] == "V": epoch_x = epoch_x + "31.0/"

    # year
    if epoch[0:1] == "I": epoch_x = epoch_x + "18"
    elif epoch[0:1] == "J": epoch_x = epoch_x + "19"
    elif epoch[0:1] == "K": epoch_x = epoch_x + "20"
    epoch_x = epoch_x + epoch[1:3]

    if unc_U == "": unc_U = "?"
    expanded_designation = readable_designation + " " + unc_U

    # Write XEphem format orbit to the full_catalog list.
    full_catalog.append(expanded_designation + ",e," + inclin + "," 
    + node + "," + peri + "," + a + "," + motion_n + "," + eccen + "," + 
    mean_anomaly_M + "," + epoch_x + "," + "2000" + ",H " + abs_m_H + 
    "," + slope_G + "\n")

#f2 = file("Monthly_Orbit_Catalogs/" + current_year + "_" + current_month + 
#"_ORB.DAT", "w")

f2 = open(current_year + "_" + current_month + 
"_ORB.DAT", "w")
for obj in full_catalog:
    f2.write(obj)
f2.close()



    
    
    
    
    
    
    
    
    
    
    
    
