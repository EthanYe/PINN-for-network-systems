import numpy as np
import matplotlib.pyplot as plt

name1 = 'seven_pipe_1_reach'
name3 = 'seven_pipe_3_reach'
name5 = 'seven_pipe_5_reach'
name_moc = 'moc_simulation_results'
t1 = 5
tlen = 4000
fignum = 1
fig1=plt.figure(fignum, figsize=(10, 5))
t = np.linspace(0, t1, tlen)
y0 = np.load(name_moc + '.npy')
y1 = np.load(name1 + '.npy')
y3 = np.load(name3 + '.npy')
y5 = np.load(name5 + '.npy')
errors1=[]
errors3=[]
errors5=[]
for j, i in enumerate([0,3]):
    plt.subplot(2, 1, j + 1)
    plt.plot(t, y0[ 7+i],'black',linewidth=1.2,label="MOC", alpha=0.8)
    plt.plot(t, y1[7 + i], '--', label="1-element EWC", alpha=0.8)
    plt.plot(t, y3[7 * 3 + i], '-.', label="3-element EWC", alpha=0.8)
    plt.plot(t, y5[7 * 5 + i], ':', label="5-element EWC", alpha=0.8)
    error1=np.sqrt(np.mean((y1[7 + i]-y0[7+i])**2/y0[7+i]**2))*100
    error3=np.sqrt(np.mean((y3[7*3 + i]-y0[7+i])**2/y0[7+i]**2))*100
    error5=np.sqrt(np.mean((y5[7*5 + i]-y0[7+i])**2/y0[7+i]**2))*100
    errors1.append(error1)
    errors3.append(error3)
    errors5.append(error5)
    plt.xlabel(('Time (s)'),fontsize=12,)
    plt.ylabel(('Head (m)'), fontsize=12,)
    if j==0:
        plt.legend(ncol=4, loc=4, fontsize=12, bbox_to_anchor=(1, 1.08))
    plt.xlim([0,5])
plt.subplots_adjust(wspace=0.3, hspace=0.5)

fig1.text(0.06, 0.9, '(a)', fontsize=12)
fig1.text(0.06, 0.46, '(b)', fontsize=12)
print(np.mean(errors1), np.mean(errors3), np.mean(errors5))  
fig1.savefig("Comparison of different reaches head.tif", dpi=300)
plt.show()
