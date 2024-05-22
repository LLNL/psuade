import os
import numpy as np 
import matplotlib.pyplot as plt

plt.figure(figsize=(30,25))
ll=np.linspace(1.6,7.2,8)

f_name="MeasurementData"
snap_t = np.genfromtxt(f_name);
snap_i = snap_t[0:2025]
plt.subplot(5,5,25)
plt.contourf(snap_i.reshape(45,45))#,cmap='Greys')
plt.colorbar(ticks=ll)
plt.xlabel('X',)
plt.ylabel('Y')
plt.title('True Solution')
plt.axis('off')

f_name='PosteriorX'
data_i=np.genfromtxt(f_name);
nsize = np.shape(data_i)[0]
likelihood = np.zeros(nsize)
for ii in range(nsize):
  likelihood[ii] = data_i[ii,4050]

f_name='PosteriorY'
data_i=np.genfromtxt(f_name);

nsize2 = nsize
if nsize > 1000:
  nsize2 = 1000

step = nsize2 / 20

# the first 5
kk = 1
for ii in range(5):
   rsnap_i = data_i[ii,0:2025]
   plt.subplot(5,5,kk)
   plt.contourf(rsnap_i.reshape(45,45))#,cmap='Greys')
   plt.colorbar(ticks=ll)
   plt.xlabel('X',)
   plt.ylabel('Y')
   plt.axis('off')
   kk = kk + 1
   print("Sample " + str(ii+1) + ": likelihood = " + str(likelihood[ii]))

# the next 5 one-fourth away
for ii in range(5):
   rsnap_i = data_i[ii*step+nsize2/4,0:2025]
   plt.subplot(5,5,kk)
   plt.contourf(rsnap_i.reshape(45,45))#,cmap='Greys')
   plt.colorbar(ticks=ll)
   plt.xlabel('X',)
   plt.ylabel('Y')
   plt.axis('off')
   kk = kk + 1
   jj = int(ii*step+nsize2/4)
   print("Sample " + str(jj+1) + ": likelihood = " + str(likelihood[jj]))

# the next 5 one-half away
for ii in range(5):
   rsnap_i = data_i[ii*step+2*nsize2/4,0:2025]
   plt.subplot(5,5,kk)
   plt.contourf(rsnap_i.reshape(45,45))#,cmap='Greys')
   plt.colorbar(ticks=ll)
   plt.xlabel('X',)
   plt.ylabel('Y')
   plt.axis('off')
   kk = kk + 1
   jj = int(ii*step+2*nsize2/4)
   print("Sample " + str(jj+1) + ": likelihood = " + str(likelihood[jj]))

# the next 5 three-quarter away
for ii in range(5):
   rsnap_i = data_i[ii*step+3*nsize2/4,0:2025]
   plt.subplot(5,5,kk)
   plt.contourf(rsnap_i.reshape(45,45))#,cmap='Greys')
   plt.colorbar(ticks=ll)
   plt.xlabel('X',)
   plt.ylabel('Y')
   plt.axis('off')
   kk = kk + 1
   jj = int(ii*step+3*nsize2/4)
   print("Sample " + str(jj+1) + ": likelihood = " + str(likelihood[jj]))

# the best 4 
kk = kk + 3
for ii in range(4):
   minVal = 1000000000
   minInd = -1
   for jj in range(nsize):
     if (likelihood[jj] < minVal):
        minVal = likelihood[jj]
        minInd = jj
   rsnap_i = data_i[minInd,0:2025]
   plt.subplot(5,5,kk)
   plt.contourf(rsnap_i.reshape(45,45))#,cmap='Greys')
   plt.colorbar(ticks=ll)
   plt.xlabel('X',)
   plt.ylabel('Y')
   plt.axis('off')
   kk = kk - 1
   print("Sample " + str(minInd+1) + ": likelihood = " + str(minVal))
   likelihood[minInd] = 100000000 

plt.show()
plt.savefig('MatplotlibMCMCPostY.png')
print('No plot, but gimp MatplotlibMCMCPostY.png instead.')


