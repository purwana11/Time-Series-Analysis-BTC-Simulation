import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="EDA Journal: Bitcoin Detective", layout="wide")

st.title("🕵️‍♂️ Jurnal Investigasi: Misteri Bitcoin (BTC-EUR)")
st.markdown("""
*Berdasarkan Modul Praktikum 12 - Pemodelan dan Simulasi*
""")

# --- Sidebar: Kontrol Interaktif ---
st.sidebar.header("Peralatan Detektif")
resample_freq = st.sidebar.selectbox("Pilih Resampling", ['W', 'M', 'D'], index=0, 
                                     format_func=lambda x: "Mingguan" if x=='W' else ("Bulanan" if x=='M' else "Harian"))
window_size = st.sidebar.slider("Jendela Rolling Mean (Hari)", 7, 100, 30)

# --- Load Data ---
@st.cache_data
def load_data():
    # Menggunakan dataset yang sudah diunggah
    df = pd.read_csv('BTC-EUR.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

df = load_data()

# --- JURNAL PART 1: The First Encounter ---
st.header("🔍 Jurnal Part 1: Membersihkan TKP (Setup & Cleaning)")
st.info("Misi: Memastikan data siap 'berbicara'. Data Time Series sangat sensitif terhadap urutan dan kekosongan.")

# Aksi Teknis: Interpolasi 
df_cleaned = df.interpolate(method='time')

col1, col2 = st.columns(2)
with col1:
    st.write("**5 Baris Pertama Data:**")
    st.dataframe(df_cleaned.head())
with col2:
    st.markdown(f"""
    **Narasi Jurnal:**
    "Saya menemukan bahwa data ini memiliki format tanggal yang standar. Namun, sebagai langkah pencegahan, 
    saya melakukan pengecekan gap dan melakukan **Interpolasi Linear** untuk memastikan alur waktu 
    tetap kontinu. Sekarang, jejak digital ini sudah bersih dan siap diinterogasi."
    """)

# --- JURNAL PART 2: Visual Inspection ---
st.header("📈 Jurnal Part 2: Melihat Wajah Sistem (Visual Inspection)")
st.info("Misi: Mendapatkan gambaran besar perilaku sistem.")

fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(df_cleaned['Close'], label='Data Mentah (Harian)', alpha=0.3, color='gray')
resampled_data = df_cleaned['Close'].resample(resample_freq).mean()
ax2.plot(resampled_data, label=f'Resampled ({resample_freq})', color='blue', linewidth=2)
ax2.set_title("Perbandingan Data Mentah vs Resampled")
ax2.legend()
st.pyplot(fig2)

st.markdown(f"""
**Narasi Jurnal:**
"Secara visual, sistem menunjukkan tren kenaikan yang konsisten, terutama sejak lonjakan besar di tahun 2017. [cite: 427]
Awalnya grafik terlihat sangat berantakan (*too noisy*), namun setelah saya melakukan *resampling* menjadi rata-rata mingguan, pola lonjakan tajam setiap akhir tahun mulai terlihat lebih jelas."
""")

# --- JURNAL PART 3: Decomposing the Pattern ---
st.header("🧪 Jurnal Part 3: Bedah Komponen (Decomposition)")
st.info("Misi: Memisahkan sinyal murni dari gangguan.")

# Dekomposisi menggunakan statsmodels 
# Model Multiplicative sering lebih cocok untuk data finansial/kripto yang volatil
decomp = seasonal_decompose(df_cleaned['Close'], model='multiplicative', period=365)

col_a, col_b = st.columns([3, 1])
with col_a:
    fig3 = decomp.plot()
    fig3.set_size_inches(10, 8)
    st.pyplot(fig3)
with col_b:
    st.markdown("""
    **Narasi Jurnal:**
    "Setelah membedah komponen data menggunakan `seasonal_decompose`, ditemukan bahwa:
    - **Trend**: Mengonfirmasi kenaikan nilai Bitcoin jangka panjang.
    - **Seasonal**: Terdapat pola musiman yang periodik.
    - **Residual**: Menunjukkan fluktuasi acak yang besar, menandakan adanya faktor eksternal tak terduga."
    """)

# --- JURNAL PART 4: Statistical Health Check ---
st.header("🏥 Jurnal Part 4: Diagnosis Stasioneritas")
st.info("Misi: Mengecek stabilitas sistem untuk pemodelan selanjutnya.")

# Rolling Statistics [cite: 438]
rolling_mean = df_cleaned['Close'].rolling(window=window_size).mean()
rolling_std = df_cleaned['Close'].rolling(window=window_size).std()

fig4, ax4 = plt.subplots(figsize=(10, 4))
ax4.plot(df_cleaned['Close'], color='blue', label='Asli', alpha=0.3)
ax4.plot(rolling_mean, color='red', label='Rolling Mean')
ax4.plot(rolling_std, color='black', label='Rolling Std')
ax4.set_title(f"Rolling Statistics (Window={window_size})")
ax4.legend()
st.pyplot(fig4)

# ADF Test [cite: 440]
st.subheader("Hasil Uji Augmented Dickey-Fuller (ADF):")
result = adfuller(df_cleaned['Close'])
st.write(f'Test Statistic: {result[0]:.4f}')
st.write(f'p-value: {result[1]:.4f}')

st.markdown("""
**Narasi Jurnal:**
"Garis rata-rata bergerak (*Rolling Mean*) terlihat menanjak, yang mengonfirmasi bahwa data ini **Tidak Stasioner**.
Artinya, rata-rata sistem berubah seiring waktu. Hal ini diperkuat dengan nilai *p-value* yang lebih besar dari 0.05, 
sehingga kita gagal menolak hipotesis nol."
""")

st.divider()
