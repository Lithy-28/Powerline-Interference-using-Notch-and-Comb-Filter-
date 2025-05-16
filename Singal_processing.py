import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz, lfilter, iirnotch, butter, filtfilt
from scipy import signal
from scipy.signal import butter
import pandas as pd
def plot_comb_filter_response(zeros, sampling_rate):
  if sampling_rate==1000:
    plt.figure()
    plt.scatter(np.real(zeros), np.imag(zeros), marker='x', color='r', label='Zeros')
    plt.title('Zeros of Comb Filter in Z-Plane fs 1000')
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.legend()
    plt.grid(True)
    plt.show()
  elif sampling_rate==1500:
    plt.figure()
    plt.scatter(np.real(zeros), np.imag(zeros), marker='x', color='r', label='Zeros')
    plt.title('Zeros of Comb Filter in Z-Plane fs 1500')
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.legend()
    plt.grid(True)
    plt.show()
  elif sampling_rate==500:
    plt.figure()
    plt.scatter(np.real(zeros), np.imag(zeros), marker='x', color='r', label='Zeros')
    plt.title('Zeros of Comb Filter in Z-Plane fs 500')
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.legend()
    plt.grid(True)
    plt.show()
def comb_filter_frequency_response(b, a, sampling_rate):
    
    w, h = freqz(b, a, worN=8000, fs=sampling_rate)
    if sampling_rate==1000:
     plt.figure()
     plt.plot(w, 20 * np.log10(np.abs(h)))
     plt.title('Magnitude Response of Comb Filter for sampling rate of 1000')
     plt.xlabel('Frequency (Hz)')
     plt.ylabel('Magnitude (dB)')
     plt.ylim(-60, 20) 
     plt.grid(True)
     
     plt.show()
     plt.figure()
     plt.plot(w, np.angle(h))
     plt.title('Phase Response of Comb Filter for sampling rate of 1000 ')
     plt.xlabel('Frequency (Hz)')
     plt.ylabel('Phase (radians)')
     plt.grid(True)
     plt.xlim(0, 400)
   
     plt.show()
    elif sampling_rate==1500:
     plt.figure()
     plt.plot(w, 20 * np.log10(np.abs(h)))
     plt.title('Magnitude Response of Comb Filter for sampling rate of 1500')
     plt.xlabel('Frequency (Hz)')
     plt.ylabel('Magnitude (dB)')
     plt.ylim(-60, 20)
     plt.grid(True)
     plt.show()
   
     plt.figure()
     plt.plot(w, np.angle(h))
     plt.title('Phase Response of Comb Filter for sampling rate of 1500 ')
     plt.xlabel('Frequency (Hz)')
     plt.ylabel('Phase (radians)')
     plt.grid(True)
     plt.xlim(0, 400)
   
     plt.show() 
    elif sampling_rate==500:
     plt.figure()
     plt.plot(w, 20 * np.log10(np.abs(h)))
     plt.title('Magnitude Response of Comb Filter for sampling rate of 500')
     plt.xlabel('Frequency (Hz)')
     plt.ylabel('Magnitude (dB)')
     plt.ylim(-60, 20) 
     plt.grid(True)
   
     plt.show()
     plt.figure()
     plt.plot(w, np.angle(h))
     plt.title('Phase Response of Comb Filter for sampling rate of 500 ')
     plt.xlabel('Frequency (Hz)')
     plt.ylabel('Phase (radians)')
     plt.grid(True)
     plt.xlim(0, 300)
   
     plt.show() 
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    
   
    if len(data) <= order:
        raise ValueError("The length of the input vector must be greater than the filter order.")
    
    filtered_data = filtfilt(b, a, data)
    return filtered_data

csv_file_path = r"C:\Users\LITHI\Desktop\sp\100.csv"  
csv_file_path_y = r"C:\Users\LITHI\Desktop\sp\100.csv"
your_sampling_rate = 1000  

ecg_data = pd.read_csv(csv_file_path).iloc[0:2000,0].values 
ecg_data_y = pd.read_csv(csv_file_path_y).iloc[0:2000,0].values

lowcut = 2.0  
highcut = 50.0  

filtered_ecg = bandpass_filter(ecg_data, lowcut, highcut, your_sampling_rate)
filtered_ecg_y = bandpass_filter(ecg_data_y, lowcut, highcut, your_sampling_rate)



plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(ecg_data, label='Original ECG')
plt.title('Original ECG Signal')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.legend()


plt.subplot(2, 1, 2)
plt.plot(filtered_ecg, label='Filtered ECG (2-50 Hz)')
plt.title('Filtered ECG Signal (2-50 Hz)')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.legend()

plt.tight_layout()  
plt.show()


plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(ecg_data_y, label='Original 2nd ECG')
plt.title('Original ECG Signal')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.legend()


plt.subplot(2, 1, 2)
plt.plot(filtered_ecg_y, label='Filtered ECG (2-50 Hz)')
plt.title('Filtered ECG Signal (2-50 Hz)')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.legend()

plt.tight_layout() 
plt.show()

ecg_data = np.loadtxt('ECG.txt', delimiter=',')

noisy_ecg_data = ecg_data + 0.2 * np.random.randn(len(ecg_data), ecg_data.shape[1])
z1=[]
z2=[]
z12=[]
z22=[]
z13=[]
z23=[]

for i in range(0,8) :
   if i%2!=0:
     z1.append((np.cos((np.pi/8.33)*i)+1j*np.sin((np.pi/8.33)*i)))
     z2.append((np.cos((np.pi/8.33)*i)-1j*np.sin((np.pi/8.33)*i)))

     z12.append((np.cos((np.pi/12.5)*i)+1j*np.sin((np.pi/12.5)*i)))
     z22.append((np.cos((np.pi/12.5)*i)-1j*np.sin((np.pi/12.5)*i)))

     z13.append((np.cos((np.pi/4.17)*i)+1j*np.sin((np.pi/4.17)*i)))
     z23.append((np.cos((np.pi/4.17)*i)-1j*np.sin((np.pi/4.17)*i)))     
zeros=z1
zeros1=z12
zeros2=z13     
zeroes=z1
# Comb filter coefficients
b1=[0.65376569,-2.3786447,4.85001463,-7.07101701,7.96483996, -7.07101701,4.85001463, -2.3786447,   0.65376569]#for sampling period of 1500
b=[0.6310, -0.2149, 0.1512, -0.1288, 0.1227, -0.1288, 0.1512, -0.2149, 0.6310] #for sampling period of 1000
# Comb filter coefficients
b2 = [3.20512821 ,1.18700923 ,1.00696344 ,1.28091747 ,5.87026854, 1.28091747,1.00696344, 1.18700923, 3.20512821]#for sampling period of 500
a = [1]


def notch_filter_frequency_response(zeros,fs):
   if fs == 1000:
  
    b = np.poly(zeros)
    a = [1] + [0] * (len(zeros) - 1)

    b_normalized = np.array(b) / np.abs(np.polyval(b, 1))


    frequencies, response = signal.freqz(b_normalized, a, fs=fs)

    
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(frequencies, 20 * np.log10(np.abs(response)))
    plt.title('Magnitude Response of the Notch Filter fs=1000')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude Response (dB)')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(frequencies, np.angle(response, deg=True))
    plt.title('Phase Response of the Notch Filter fs=1000')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (degrees)')
    
    plt.grid(True)

    plt.tight_layout()
    plt.show()
   elif fs==1500:
     b = np.poly(zeros)
     a = [1] + [0] * (len(zeros) - 1)

     b_normalized = np.array(b) / np.abs(np.polyval(b, 1))

     #
     frequencies, response = signal.freqz(b_normalized, a, fs=fs)


     plt.figure(figsize=(12, 6))

     plt.subplot(2, 1, 1)
     plt.plot(frequencies, 20 * np.log10(np.abs(response)))
     plt.title('Magnitude Response of the Notch Filter fs=1500')
     plt.xlabel('Frequency (Hz)')
     plt.ylabel('Magnitude Response (dB)')
     plt.grid(True)

     plt.subplot(2, 1, 2)
     plt.plot(frequencies, np.angle(response, deg=True))
     plt.title('Phase Response of the Notch Filter fs=1500')
     plt.xlabel('Frequency (Hz)')
     plt.ylabel('Phase (degrees)')
     
     plt.grid(True)

     plt.tight_layout()
     plt.show()
   elif fs==500:
      b = np.poly(zeros)
      a = [1] + [0] * (len(zeros) - 1)

      
      b_normalized = np.array(b) / np.abs(np.polyval(b, 1))

      
      frequencies, response = signal.freqz(b_normalized, a, fs=fs)

      
      plt.figure(figsize=(12, 6))

      plt.subplot(2, 1, 1)
      plt.plot(frequencies, 20 * np.log10(np.abs(response)))
      plt.title('Magnitude Response of the Notch Filter fs=500' )
      plt.xlabel('Frequency (Hz)')
      plt.ylabel('Magnitude Response (dB)')
      plt.grid(True)

      plt.subplot(2, 1, 2)
      plt.plot(frequencies, np.angle(response, deg=True))
      plt.title('Phase Response of the Notch Filter fs=500')
      plt.xlabel('Frequency (Hz)')
      plt.ylabel('Phase (degrees)')
      
      plt.grid(True)

      plt.tight_layout()
      plt.show()
fs=1000
fs1=1500
fs2=500

zero=[0.9+0.3j,0.9-0.3j]
notch_filter_frequency_response(zero,fs)
zero1=[0.9685+0.2896j,0.9685-0.2986j]
notch_filter_frequency_response(zero1,fs1)
zero2 = [0.7 + 0.6j, 0.7 - 0.6j]
notch_filter_frequency_response(zero2,fs2)

# Sampling rate
sampling_rate = 1000
sampling_rate1=1500
sampling_rate2=500

zeros=z1
zeros1=z12
zeros2=z13
  
plot_comb_filter_response(zeros, sampling_rate)
comb_filter_frequency_response(b, a, sampling_rate)

plot_comb_filter_response(zeros1, sampling_rate1)
comb_filter_frequency_response(b1, a, sampling_rate1)

plot_comb_filter_response(zeros2, sampling_rate2)
comb_filter_frequency_response(b2, a, sampling_rate2)






