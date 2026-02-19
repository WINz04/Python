import wx
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv
from sklearn.metrics import r2_score, mean_absolute_error

# Untuk Mengatur Style Grafik
sns.set(style="darkgrid")


# Tentukan Nama Aplikasi
class MyApp(wx.App):
    def OnInit(self):
        frame = MyFrame(None, title='Prediksi Jumlah Daging Potong Pada Rumah Potong Hewan (RPH)')
        frame.Show()
        return True


class MyFrame(wx.Frame):
    def __init__(self, parent, title):
        super().__init__(parent, title=title, size=(900, 700))

        self.data_jenis_daging = pd.DataFrame()  # Data jenis daging yang sudah diimpor
        self.SetBackgroundColour(wx.Colour(200, 200, 200))
        panel = wx.Panel(self)

        #Pembuatan Tombol dan Menu Dropdown

        jenis_daging_choices = ['Sapi', 'Kerbau', 'Kuda', 'Kambing', 'Domba', 'Babi']
        self.jenis_daging_dropdown = wx.ComboBox(panel, choices=jenis_daging_choices, style=wx.CB_DROPDOWN)
        self.jenis_daging_dropdown.Disable()  # Menonaktifkan dropdown jenis daging

        provinsi_choices = self.load_provinsi_choices()
        self.provinsi_dropdown = wx.ComboBox(panel, choices=provinsi_choices, style=wx.CB_DROPDOWN)
        self.provinsi_dropdown.Disable()  # Menonaktifkan dropdown provinsi

        self.import_button = wx.Button(panel, label='Import Data')
        self.import_button.SetBackgroundColour(wx.Colour(0, 200, 0))

        self.tampilkan_data_button = wx.Button(panel, label='Tampilkan Data Asli')
        self.tampilkan_data_button.SetBackgroundColour(wx.Colour(0, 255, 255))

        self.train_button = wx.Button(panel, label='Hasil Prediksi')
        self.train_button.SetBackgroundColour(wx.Colour(0, 255, 255))

        self.hapus_button = wx.Button(panel, label='Hapus')
        self.hapus_button.SetBackgroundColour(wx.Colour(255, 0, 0))

        self.download_button = wx.Button(panel, label='Download Prediksi')
        self.download_button.SetBackgroundColour(wx.Colour(0, 200, 0))

        self.akurasi_button = wx.Button(panel, label='Akurasi')
        self.akurasi_button.SetBackgroundColour(wx.Colour(0, 200, 0))
        self.akurasi_input = wx.TextCtrl(panel)
        self.output_text = wx.TextCtrl(panel, style=wx.TE_MULTILINE)
        self.graph_output = wx.StaticBitmap(panel)

        # Buat layout
        vbox = wx.BoxSizer(wx.VERTICAL)
        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        hbox1.Add(wx.StaticText(panel, label='Jenis Daging:'), flag=wx.ALIGN_CENTER_VERTICAL | wx.ALL, border=5)
        hbox1.Add(self.jenis_daging_dropdown, flag=wx.ALL, border=5)
        hbox1.Add(wx.StaticText(panel, label='Provinsi:'), flag=wx.ALIGN_CENTER_VERTICAL | wx.ALL, border=15)
        hbox1.Add(self.provinsi_dropdown, flag=wx.ALL, border=5)
        hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        hbox2.Add(self.import_button, flag=wx.ALL, border=5)
        hbox2.Add(self.tampilkan_data_button, flag=wx.ALL, border=5)
        hbox2.Add(self.train_button, flag=wx.ALL, border=5)
        hbox2.Add(self.hapus_button, flag=wx.ALL, border=5)
        hbox2.Add(self.download_button, flag=wx.ALL, border=5)
        hbox2.Add(self.akurasi_button, flag=wx.ALL, border=5)
        vbox.Add(hbox1, flag=wx.EXPAND | wx.ALL, border=10)
        vbox.Add(hbox2, flag=wx.EXPAND | wx.ALL, border=10)
        vbox.Add(wx.StaticText(panel, label='Input Data Daging : '), flag=wx.ALL, border=10)
        vbox.Add(self.akurasi_input, flag=wx.ALL, border=10)
        vbox.Add(wx.StaticText(panel, label='Prediksi Jumlah Daging Potong 2021 : '), flag=wx.ALL, border=10)
        vbox.Add(self.output_text, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)
        vbox.Add(self.graph_output, proportion=1, flag=wx.EXPAND | wx.ALL, border=10)

        # Fungsi Ketika Tombol Di klik
        self.import_button.Bind(wx.EVT_BUTTON, self.import_data)
        self.tampilkan_data_button.Bind(wx.EVT_BUTTON, self.tampilkan_data_asli)
        self.train_button.Bind(wx.EVT_BUTTON, self.train_and_predict_model)
        self.hapus_button.Bind(wx.EVT_BUTTON, self.reset_data)
        self.download_button.Bind(wx.EVT_BUTTON, self.download_prediction)
        self.akurasi_button.Bind(wx.EVT_BUTTON, self.calculate_accuracy)

        # Set keadaan Tombol Ketika Aplikasi Berjalan
        self.tampilkan_data_button.Disable()
        self.train_button.Disable()
        self.download_button.Disable()
        self.akurasi_button.Disable()

        # Mengatur tata letak (layout)
        panel.SetSizer(vbox)

    # Pilih Provinsi
    def load_provinsi_choices(self):
        if os.path.exists('data.csv'):
            data = pd.read_csv('data.csv')
            self.data_jenis_daging = data.copy()
            self.jenis_daging_dropdown.Enable()  # Mengaktifkan dropdown jenis daging
            return data['provinsi'].unique().tolist()
        return []

    # FUnction Untuk Input Data Excel
    def import_data(self, event):
        wildcard = "CSV files (*.csv)|*.csv"
        dialog = wx.FileDialog(None, "Import Data", wildcard=wildcard, style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        if dialog.ShowModal() == wx.ID_OK:
            filename = dialog.GetPath()
            data = pd.read_csv(filename)
            self.data_jenis_daging = data.copy()
            data.to_csv('data.csv', index=False)
            provinsi_choices = data['provinsi'].unique().tolist()
            self.provinsi_dropdown.SetItems(provinsi_choices)
            self.provinsi_dropdown.Enable()  # Mengaktifkan dropdown provinsi
            self.provinsi_dropdown.SetValue('')
            self.jenis_daging_dropdown.Enable()  # Mengaktifkan dropdown jenis daging
            self.tampilkan_data_button.Enable()
            self.train_button.Enable()

        dialog.Destroy()

    # Funtion untuk Tombol Data Asli
    def tampilkan_data_asli(self, event):
        self.reset_output()

        #Pilih Jenis Daging dan Provinsi
        jenis_daging = self.jenis_daging_dropdown.GetValue()
        provinsi = self.provinsi_dropdown.GetValue()

        if not jenis_daging or not provinsi:
            self.output_text.AppendText("Silakan pilih jenis daging dan provinsi terlebih dahulu.\n")
            return

        if self.data_jenis_daging.empty:
            self.output_text.AppendText("Data belum diimport. Silakan import data terlebih dahulu.\n")
            return

        data_provinsi = self.data_jenis_daging[self.data_jenis_daging['provinsi'] == provinsi]
        data_asli = data_provinsi[['tahun', jenis_daging]]

        # tampilkan pada Text
        self.output_text.AppendText(f"Data Asli {jenis_daging} di {provinsi}:\n")
        self.output_text.AppendText(data_asli.to_string(index=False, justify='left', col_space=12))

        # Buat plot/Grafik
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=data_asli, x='tahun', y=jenis_daging)
        ax.set_title(f"Data Asli {jenis_daging} di {provinsi}")
        ax.set_xlabel('Tahun')
        ax.set_ylabel(f'Jumlah Pemotongan {jenis_daging}')

        # Simpan Gambar
        plt.savefig('data_asli_plot.png')

        # memuat gambar plot/Grafik dan menampilkannya
        image = wx.Image('data_asli_plot.png', wx.BITMAP_TYPE_ANY)
        image = image.Scale(500, 300, wx.IMAGE_QUALITY_HIGH)
        bitmap = wx.Bitmap(image)
        self.graph_output.SetBitmap(bitmap)

    # Function Untuk Latihan Jaringan syaraf tiruan Menggunakan Metode Backpropagation Pendekaran Regresi MLPRegressor dan MinMaxScaler
    def train_and_predict_model(self, event):
        self.reset_output()

        jenis_daging = self.jenis_daging_dropdown.GetValue()
        provinsi = self.provinsi_dropdown.GetValue()

        if not jenis_daging or not provinsi:
            self.output_text.AppendText("Silakan pilih jenis daging dan provinsi terlebih dahulu.\n")
            return

        if self.data_jenis_daging.empty:
            self.output_text.AppendText("Data belum diimport. Silakan import data terlebih dahulu.\n")
            return

        data_provinsi = self.data_jenis_daging[self.data_jenis_daging['provinsi'] == provinsi]

        # Mempersiapkan Data
        X = data_provinsi['tahun'].values.reshape(-1, 1)
        y = data_provinsi[jenis_daging].values

        # Mengubah Nilai X ke dalam rentang [0,1]
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X) # Fungsi MinMax Scaler pada X

        # Regresi MLPRegressor
        model = MLPRegressor(hidden_layer_sizes=(100, 200), activation='relu', solver='adam', tol=1e-7, max_iter=100000,
                             random_state=42)

        # Pelatihan Jaringan Syaraf Tiruan
        model.fit(X, y)

        X_2021 = scaler.transform([[2021]])  #Membuat Array 2D Sebagai Input prediksi
        pred_2021 = model.predict(X_2021)[0]

        # menampilkan teks pada pada saat output
        self.output_text.AppendText(
            f"Prediksi Jumlah Pemotongan {jenis_daging} Tahun 2021 di {provinsi}: {pred_2021:.1f}\n")

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=data_provinsi, x='tahun', y=jenis_daging, label='Data Asli')
        sns.scatterplot(x=[2021], y=[pred_2021], color='red', marker='o', s=100, label='Prediksi 2021')
        ax.set_title(f"Jumlah Pemotongan {jenis_daging} per Tahun")
        ax.set_xlabel('Tahun')
        ax.set_ylabel(f'Jumlah Pemotongan {jenis_daging}')
        ax.legend()

        # Save plot to image
        plt.savefig('plot.png')

        # Load plot image and display
        image = wx.Image('plot.png', wx.BITMAP_TYPE_ANY)
        image = image.Scale(500, 300, wx.IMAGE_QUALITY_HIGH)
        bitmap = wx.Bitmap(image)
        self.graph_output.SetBitmap(bitmap)

        self.download_button.Enable()  # Mengaktifkan tombol download
        self.akurasi_button.Enable()  # Mengaktifkan tombol akurasi

        # Simpan data asli dan hasil prediksi ke Excel
        df_asli = data_provinsi[['tahun', jenis_daging]]
        df_prediksi = pd.DataFrame({'tahun': [2021], jenis_daging: [pred_2021]})

        df_combined = pd.concat([df_asli, df_prediksi], ignore_index=True)

        df_combined.to_excel('prediksi_daging_potong.xlsx', index=False)

    # Function Reset Data
    def reset_data(self, event):
        self.reset_output()
        if os.path.exists('data.csv'):
            os.remove('data.csv')
        if os.path.exists('prediksi_daging_potong.xlsx'):
            os.remove('prediksi_daging_potong.xlsx')
        self.provinsi_dropdown.SetItems([])
        self.provinsi_dropdown.SetValue('')
        self.provinsi_dropdown.Disable()  # Menonaktifkan dropdown provinsi
        self.jenis_daging_dropdown.Disable()  # Menonaktifkan dropdown jenis daging
        self.data_jenis_daging = pd.DataFrame()
        self.tampilkan_data_button.Disable()
        self.train_button.Disable()
        self.download_button.Disable()
        self.akurasi_button.Disable()

    # Reset Output teks dan Grafik
    def reset_output(self):
        self.output_text.Clear()
        self.graph_output.SetBitmap(wx.NullBitmap)
        plt.close()

    # Tombol Donwload hasil Prediksi
    def download_prediction(self, event):
        if os.path.exists('prediksi_daging_potong.xlsx'):
            wildcard = "Excel files (*.xlsx)|*.xlsx"
            dialog = wx.FileDialog(None, "Download Prediksi", wildcard=wildcard, style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
            if dialog.ShowModal() == wx.ID_OK:
                destination = dialog.GetPath()
                os.replace('prediksi_daging_potong.xlsx', destination)
            dialog.Destroy()

    # Untuk Akurasi
    def calculate_accuracy(self, event):
        self.reset_output()

        jenis_daging = self.jenis_daging_dropdown.GetValue()
        provinsi = self.provinsi_dropdown.GetValue()

        if not jenis_daging or not provinsi:
            self.output_text.AppendText("Silakan pilih jenis daging dan provinsi terlebih dahulu.\n")
            return

        if self.data_jenis_daging.empty:
            self.output_text.AppendText("Data belum diimport. Silakan import data terlebih dahulu.\n")
            return

        data_provinsi = self.data_jenis_daging[self.data_jenis_daging['provinsi'] == provinsi]

        X = data_provinsi['tahun'].values.reshape(-1, 1)
        y = data_provinsi[jenis_daging].values

        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

        model = MLPRegressor(hidden_layer_sizes=(100, 200), activation='relu', solver='adam', tol=1e-7, max_iter=100000,
                             random_state=42)
        model.fit(X, y)

        X_2021 = scaler.transform([[2021]])
        pred_2021 = model.predict(X_2021)[0]

        self.output_text.AppendText(
            f"Prediksi Jumlah Pemotongan {jenis_daging} Tahun 2021 di {provinsi}: {pred_2021:.1f}\n")

        # R^2 Score
        akurasi_input = self.akurasi_input.GetValue()

        if not akurasi_input:
            self.output_text.AppendText("Silakan masukkan nilai aktual untuk menghitung akurasi.\n")
            return

        try:
            akurasi_input = float(akurasi_input)
        except ValueError:
            self.output_text.AppendText("Nilai aktual yang dimasukkan tidak valid.\n")
            return

        # Nilai MAE
        if akurasi_input != 0:
            akurasi = 1 - abs(akurasi_input - pred_2021) / akurasi_input
        else:
            # Handle jika akurasi_input adalah nol
            akurasi = 0  # Atau nilai lain yang sesuai untuk mengatasi kasus ini
        self.output_text.AppendText(f"Akurasi: {akurasi:.2%}\n")

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=data_provinsi, x='tahun', y=jenis_daging, label='Data Asli')
        sns.scatterplot(x=[2021], y=[pred_2021], color='red', marker='o', s=100, label='Prediksi 2021')
        ax.set_title(f"Jumlah Pemotongan {jenis_daging} per Tahun")
        ax.set_xlabel('Tahun')
        ax.set_ylabel(f'Jumlah Pemotongan {jenis_daging}')
        ax.axhline(y=akurasi_input, color='green', linestyle='--', label='Nilai Aktual')
        ax.legend()

        # Save plot to image
        plt.savefig('akurasi_plot.png')

        # Load plot image and display
        image = wx.Image('akurasi_plot.png', wx.BITMAP_TYPE_ANY)
        image = image.Scale(500, 300, wx.IMAGE_QUALITY_HIGH)
        bitmap = wx.Bitmap(image)
        self.graph_output.SetBitmap(bitmap)


if __name__ == '__main__':
    app = MyApp()
    app.MainLoop()
