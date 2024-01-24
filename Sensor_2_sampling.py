# Sampling
import numpy as np

N = 900
Dt = 2E-4
start = 0
stop = start + N * Dt

t = np.linspace(start, stop, N)
func = lambda t : np.sin(2 * np.pi * 100 * t) 
x = func(t)
f_s = 1 / Dt

# FFT
N_FFT = 1024
Df = 1 / (N_FFT * Dt)
F_start = 0
F_stop = 1 / Dt
f = np.linspace(F_start, F_stop, N_FFT)
X = np.fft.fft(x, N_FFT)
X_mag = np.abs(X)
X_phase = np.angle(X)

# PDS
Sxx = np.abs(X)**2
Sxx_log = 20 * np.log10(np.abs(X))
Sxx_norm = Sxx_log - np.max(Sxx_log)

# Zero padding
N_ZP = 4096
Df_ZP = 1 / (N_ZP * Dt)
F_start_ZP = 0
F_stop_ZP = 1 / Dt
f_ZP = np.linspace(F_start_ZP, F_stop_ZP, N_ZP)
X_ZP = np.fft.fft(x, N_ZP)
X_mag_ZP = np.abs(X_ZP)
X_phase_ZP = np.angle(X_ZP)

# Zero padding * 2
N_ZP_2 = 8192
Df_ZP_2 = 1 / (N_ZP_2 * Dt)
F_start_ZP_2 = 0
F_stop_ZP_2 = 1 / Dt
f_ZP_2 = np.linspace(F_start_ZP_2, F_stop_ZP_2, N_ZP_2)
X_ZP_2 = np.fft.fft(x, N_ZP_2)
X_mag_ZP_2 = np.abs(X_ZP_2)
X_phase_ZP_2 = np.angle(X_ZP_2)

# Hanning window
hanning = np.hanning(N)
x_hanning = x * hanning
N_H = 1024
Df_H = 1 / (N_H * Dt)
F_start_H = 0
F_stop_H = 1 / Dt
f_H = np.linspace(F_start_H, F_stop_H, N_H)
X_H = np.fft.fft(x_hanning, N_H)
X_mag_H = np.abs(X_H)
X_phase_H = np.angle(X_H)

# Complex sinussoide
x_complex = np.exp(-1j * 2 * np.pi * 100 * t)
X_complex = np.fft.fft(x_complex, N_FFT)
X_complex_mag = np.abs(X_complex)
X_complex_phase = np.angle(X_complex)

# Complex sinussoide shift
X_complex_shift = np.fft.fftshift(X_complex)
X_complex_shift_mag = np.abs(X_complex_shift)
X_complex_shift_phase = np.angle(X_complex_shift)
f_shift = np.linspace(-F_stop/2, F_stop/2, N_FFT)

# Complex sinussoide positive
x_complex_pos = np.exp(1j * 2 * np.pi * 100 * t)
X_complex_pos = np.fft.fft(x_complex_pos, N_FFT)
X_complex_pos_mag = np.abs(X_complex_pos)
X_complex_pos_phase = np.angle(X_complex_pos)
X_complex_pos_shift = np.fft.fftshift(X_complex_pos)
X_complex_pos_shift_mag = np.abs(X_complex_pos_shift)
X_complex_pos_shift_phase = np.angle(X_complex_pos_shift)

# Plotting
import matplotlib.pyplot as plt
# -- Time domain (1) --
def one():
    plt.plot(t[0:200], x[0:200])
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude [V]')
    plt.title('x(t)')
    plt.grid()
    plt.show()


# -- Frequency domain (2a) --
def two_a():
    plt.plot(f, X_mag)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.title('X(f)')
    plt.grid()
    plt.show()

# -- Periodogram (2b) --
def two_b():
    plt.plot(f, Sxx)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.title('Sxx(f)')
    plt.grid()
    plt.show()

# -- Zoomed specter (2c) --
def two_c():
    plt.plot(f[0:41], X_mag[0:41]) # 1024/x = 5000/200 
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.title('Sxx(f)')
    plt.grid()
    plt.show()

# -- Nomralized Periodogram (2d) --
def two_d():
    plt.plot(f, Sxx_norm)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Relativ effekt [dB]')
    plt.ylim(-80, 10)
    plt.title('Sxx(f)')
    plt.grid()
    plt.show()

# -- Zero padding (3a) --
def three_a():
    plt.plot(f_ZP, X_mag_ZP)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.title('X(f)')
    plt.grid()
    plt.show()

# -- Zero padding (3b) --
def three_b():
    plt.plot(f_ZP_2, X_mag_ZP_2)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.title('X(f)')
    plt.grid()
    plt.show()

# -- Comparing -- 
def compare():
    plt.plot(f_ZP_2[0:328], X_mag_ZP_2[0:328], label='N = 8192', color='red')
    plt.plot(f_ZP[0:164], X_mag_ZP[0:164], label='N = 4096', color='green', linestyle='dashed')
    plt.plot(f[0:41], X_mag[0:41], label='N = 1024', color='blue', linestyle='dashed')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.title('X(f)')
    plt.legend()
    plt.grid()
    plt.show()

# -- Hanning window (4) --
def four():
    plt.subplot(2, 1, 2)
    plt.plot(f, X_mag, label='Uten vindu')
    plt.plot(f_H, X_mag_H, label='Med vindu')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.title('X(f)')
    plt.legend()
    plt.grid()
    plt.subplot(2, 1, 1)
    plt.plot(t, x, label='Uten vindu')
    plt.plot(t, x_hanning, label='Med vindu')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude [V]')
    plt.title('x(t)')
    plt.legend()
    plt.grid()
    plt.show()
    
# -- Complex sinussoide (5a) --
def five_a():
    plt.plot(f, X_complex_mag)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.title('X(f)')
    plt.grid()
    plt.show()
# -- Complex sinussoide shift (5b) --
def five_b():
    plt.plot(f_shift, X_complex_shift_mag)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.title('X(f)')
    plt.grid()
    plt.show()

# -- Complex sinussoide positive shift (5c) --
def five_c():
    plt.plot(f_shift, X_complex_pos_shift_mag)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.title('X(f)')
    plt.grid()
    plt.show()
# -- Menu --
menu = {}
menu['1']="Oppgave 1 Generert Sinusignal" 
menu['2']="Oppgave 2a Spektrum av signal"
menu['3']="Oppgave 2b Effekttetthetsspekter av signal"
menu['4']="Oppgave 2c Zoomed spektrum"
menu['5']="Oppgave 2d Normalized log spektrum"
menu['6']="Oppgave 3a Zero padding"
menu['7']="Oppgave 3b Zero padding * 2"
menu['8']="Oppgave 3c Sammenligning"
menu['9']="Oppgave 4 Hanning vindu"
menu['10']="Oppgave 5a Kompleks sinussoide spektrum"
menu['11']="Oppgave 5b Kompleks sinussoide shift spektrum"
menu['12']="Oppgave 5c Kompleks sinussoide positive shift spektrum"
menu['0']="Exit"
while True: 
    options=menu.keys()
    for entry in options: 
        print(entry, menu[entry])

    selection=input("Please Select: ")
    if selection =='1': 
        print("Oppgave 1" )
        one()
        break
    elif selection == '2': 
        print("Oppgave 2a")
        two_a()
        break
    elif selection == '3':
        print("Oppgave 2b") 
        two_b()
        break
    elif selection == '4':
        print("Oppgave 2c") 
        two_c()
        break
    elif selection == '5':
        print("Oppgave 2d") 
        two_d()
        break
    elif selection == '6':
        print("Oppgave 3a") 
        three_a()
        break
    elif selection == '7':
        print("Oppgave 3b") 
        three_b()
        break
    elif selection == '8':
        print("Oppgave 3c") 
        compare()
        break
    elif selection == '9':
        print("Oppgave 4") 
        four()
        break
    elif selection == '10':
        print("Oppgave 5a") 
        five_a()
        break
    elif selection == '11':
        print("Oppgave 5b") 
        five_b()
        break
    elif selection == '12':
        print("Oppgave 5c") 
        five_c()
        break
    elif selection == '0': 
        break
    else: 
        print("Unknown Option Selected!") 