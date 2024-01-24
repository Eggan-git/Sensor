# Sampling
import numpy as np

N = 900
Dt = 2E-4
start = 0
stop = start + N * Dt

t = np.linspace(start, stop, N)
func = lambda t : np.sin(2 * np.pi * 100 * t) 
x = func(t)

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


# -- Menu --
menu = {}
menu['1']="Oppgave 1" 
menu['2']="Oppgave 2a"
menu['3']="Oppgave 2b"
menu['4']="Oppgave 2c"
menu['5']="Oppgave 2d"
menu['10']="Exit"
while True: 
    options=menu.keys()
    for entry in options: 
        print(entry, menu[entry])

    selection=input("Please Select:")
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
    elif selection == '10': 
        break
    else: 
        print("Unknown Option Selected!") 