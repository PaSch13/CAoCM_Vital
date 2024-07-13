import vitaldb
import matplotlib.pyplot as plt
# %%
track_names = ['Solar8000/NIBP_DBP',
               'Solar8000/CVP',
               'Solar8000/HR',
               'Solar8000/ART_DBP',
               'Solar8000/ART_MBP',
               'Solar8000/ETCO2',
               'Solar8000/ART_SBP',
               'Solar8000/NIBP_MBP',
               'Solar8000/NIBP_SBP']

# vf = vitaldb.VitalFile(113, track_names)
vf = vitaldb.VitalFile('C:/Users/tilma/Documents/Entwicklung/CACOM/CAoCM_Vital/vital_files/demo', track_names)
vf.to_vital('vital_files/demo')

# %%
samples = vf.to_numpy(track_names, 1 / 100)
# plt.figure(figsize=(20, 5))
plt.plot(samples[:, 2])
plt.show()

# %%
import vitaldb
import matplotlib.pyplot as plt

track_names = ['SNUADC/ART']
vf = vitaldb.VitalFile(1, track_names)
samples = vf.to_numpy(track_names, 1 / 100)

plt.figure(figsize=(20, 5))
plt.plot(samples[:, 0])
plt.show()