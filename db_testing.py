import vitaldb
import matplotlib.pyplot as plt
# %%
track_names = ['Solar8000/CVP',
               'Solar8000/NIBP_DBP',
               'Solar8000/CVP',
               'Solar8000/HR',
               'Solar8000/ART_DBP',
               'Solar8000/ART_MBP',
               'Solar8000/ETCO2',
               'Solar8000/ART_SBP',
               'Solar8000/NIBP_MBP',
               'Solar8000/NIBP_SBP']

# vf = vitaldb.VitalFile(113, track_names)
<<<<<<< HEAD
vf = vitaldb.VitalFile('C:/Users/tilma/Documents/Entwicklung/CACOM/CAoCM_Vital/vital_files/demo', track_names)
vf.to_vital('vital_files/demo')
=======
# vf.to_vital('vital_files/demo3')
>>>>>>> faae65a0d22562eef0adc551a60aa1ace66ff707

samples = vitaldb.vital_recs('/Users/patrickschneider/Desktop/CAoCM/CAoCM_Vital/vital_files/demo3', track_names, 2)
plt.plot(samples[:,:])
plt.show()

num_plots = samples.shape[1]

# Create subplots
fig, axes = plt.subplots(num_plots, 1, figsize=(10, 2*num_plots))

# Plot each array in a separate subplot
for i in range(num_plots):
    axes[i].plot(samples[:, i])
    axes[i].set_title(f'Plot {track_names[i]}')
    axes[i].set_xlabel('Index')
    axes[i].set_ylabel('Value')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plots
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