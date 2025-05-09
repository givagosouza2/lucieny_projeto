import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, detrend
import numpy as np
from scipy.interpolate import interp1d


def interpolate_signal(time, data, target_fs=100):
    tamanho = len(time)

    new_time = np.arange(time[0], time[tamanho-1], 1 / target_fs)
    interp_func = interp1d(time, data, kind='linear',
                           bounds_error=False, fill_value="extrapolate")
    new_data = interp_func(new_time)
    return new_time, new_data


def process_signal(data, cutoff, fs, order=4, detrend_signal=True):
    if detrend_signal:
        data = detrend(data)

    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def plot_graph(x, y, x_label, y_label, title):

    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    return fig


# Título do aplicativo
st.title("Importar arquivo de texto com colunas numéricas")

tab1, tab2 = st.tabs(['Smartphone', 'Cinemática'])

with tab1:
    # Upload do arquivo
    uploaded_file = st.file_uploader(
        "Faça o upload de um arquivo de texto", type=["txt", "csv"])

    # Configurações do filtro
    fs = 100  # Taxa de amostragem
    cutoff = 10  # Frequência de corte (10 Hz)

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file, delimiter=";", header=0)
        if len(df.columns) >= 4:
            # Criando variáveis para cada coluna
            tempo = df.iloc[:, 0] / 1000  # Convertendo tempo para segundos
            acc_x = df.iloc[:, 1]  # Dados do eixo X
            acc_y = df.iloc[:, 2]  # Dados do eixo Y
            acc_z = df.iloc[:, 3]  # Dados do eixo Z

            # Interpolação
            tempo_interp, acc_x_interpolated = interpolate_signal(
                tempo, acc_x, target_fs=fs)
            _, acc_y_interpolated = interpolate_signal(
                tempo, acc_y, target_fs=fs)
            _, acc_z_interpolated = interpolate_signal(
                tempo, acc_z, target_fs=fs)

            # Processamento do sinal
            acc_x_processed = process_signal(
                acc_x_interpolated, cutoff=cutoff, fs=fs)
            acc_y_processed = process_signal(
                acc_y_interpolated, cutoff=cutoff, fs=fs)
            acc_z_processed = process_signal(
                acc_z_interpolated, cutoff=cutoff, fs=fs)

            fig, ax = plt.subplots()
            
            ax.plot(acc_x_processed[1500:7500], acc_z_processed[1500:7500])
            ax.set_xlabel('Acc ML')
            ax.set_ylabel("Acc AP")
            ax.set_xlim([-10, 10])
            ax.set_ylim([-10, 10])
            ax.set_aspect('equal')
            st.pyplot(fig)
            
            fig, ax = plt.subplots()
            ax.plot(tempo_interp, acc_x_processed)
            ax.set_xlabel('Tempo')
            ax.set_ylabel("Acc ML")
            ax.set_aspect('equal')
            st.pyplot(fig)

            fig, ax = plt.subplots()
            ax.plot(tempo_interp, acc_z_processed)
            ax.set_xlabel('Tempo')
            ax.set_ylabel("Acc ML")
            ax.set_aspect('equal')
            st.pyplot(fig)

            desvio_total = np.sum(
                np.sqrt(acc_x_processed[1500:7500]**2 + acc_z_processed[1500:7500]**2))

            st.write(desvio_total)
with tab2:
    # Upload do arquivo
    uploaded_file = st.file_uploader(
        "Faça o upload de um arquivo de texto da cinemática", type=["txt", "csv"])
    print('1')
    if uploaded_file is not None:
        print('2')
        df = pd.read_csv(uploaded_file, sep="\t",
                         header=0, encoding='latin1')

        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        if len(df.columns) >= 4:
            print('fui')
            # Criando variáveis para cada coluna
            # Convertendo tempo para segundos
            tempo = np.nan_to_num(df.iloc[:, 0])
            desl_x = np.nan_to_num(df.iloc[:, 1])  # Dados do eixo X
            desl_y = np.nan_to_num(df.iloc[:, 2])  # Dados do eixo Y
            desl_z = np.nan_to_num(df.iloc[:, 3])  # Dados do eixo Z
            veloc_x = np.nan_to_num(df.iloc[:, 5])  # Dados do eixo X
            veloc_y = np.nan_to_num(df.iloc[:, 6])  # Dados do eixo Y
            veloc_z = np.nan_to_num(df.iloc[:, 7])  # Dados do eixo Z
            acc_x_kinem = np.nan_to_num(df.iloc[:, 9])  # Dados do eixo X
            acc_y_kinem = np.nan_to_num(df.iloc[:, 10])  # Dados do eixo Y
            acc_z_kinem = np.nan_to_num(df.iloc[:, 11])  # Dados do eixo Z

            desl_x = detrend(desl_x)
            desl_y = detrend(desl_y)
            desl_z = detrend(desl_z)

            acc_x_kinem = detrend(acc_x_kinem)
            acc_y_kinem = detrend(acc_y_kinem)
            acc_z_kinem = detrend(acc_z_kinem)

            fig, ax = plt.subplots()
            ax.plot(desl_x, desl_y)
            ax.set_xlabel('Desl ML')
            ax.set_ylabel("Desl AP")
            ax.set_xlim([-0.5, 0.5])
            ax.set_ylim([-0.5, 0.5])
            ax.set_aspect('equal')
            st.pyplot(fig)

            fig, ax = plt.subplots()
            ax.plot(tempo, desl_x)
            ax.set_xlabel('Tempo')
            ax.set_ylabel("Desl ML")
            ax.set_aspect('equal')
            st.pyplot(fig)

            fig, ax = plt.subplots()
            ax.plot(tempo, desl_y)
            ax.set_xlabel('Tempo')
            ax.set_ylabel("Desl ML")
            ax.set_aspect('equal')
            st.pyplot(fig)
            
            deslocamento_total = np.sum(np.sqrt(desl_x**2 + desl_y**2))

            st.write(deslocamento_total)
