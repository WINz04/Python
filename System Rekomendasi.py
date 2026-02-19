# libary
import wx
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


# Aplikasi
class RecommendationApp(wx.Frame): #menggunakan WxPyhton sebagai GUI
    def __init__(self, parent, title):
        #ukurang/size dari aplikasi
        super(RecommendationApp, self).__init__(parent, title=title, size=(800, 600))

        self.panel = wx.Panel(self)
        self.vbox = wx.BoxSizer(wx.VERTICAL)

        #nama/ judul dari aplikasi sistem rekomendasi
        self.st1 = wx.StaticText(self.panel, label='Rekomendasi Tempat Wisata di Jakarta', style=wx.ALIGN_CENTER)
        font = self.st1.GetFont()
        font.PointSize += 10
        self.st1.SetFont(font)
        self.vbox.Add(self.st1, flag=wx.ALL | wx.EXPAND, border=15)
        self.hbox = wx.BoxSizer(wx.HORIZONTAL)

        #button/tombol Input data (csv)
        self.load_btn = wx.Button(self.panel, label='Input Data')
        self.hbox.Add(self.load_btn, flag=wx.EXPAND | wx.ALL, border=10)
        self.load_btn.Bind(wx.EVT_BUTTON, self.on_load)

        #button/tombol untuk tampilkan data asli tempat wisata di jakarta
        self.display_data_btn = wx.Button(self.panel, label='Tampilkan Data')
        self.hbox.Add(self.display_data_btn, flag=wx.EXPAND | wx.ALL, border=10)
        self.display_data_btn.Bind(wx.EVT_BUTTON, self.display_data)

        self.recommend_btn = wx.Button(self.panel, label='Rekomendasi')
        self.hbox.Add(self.recommend_btn, flag=wx.EXPAND | wx.ALL, border=10)
        self.recommend_btn.Bind(wx.EVT_BUTTON, self.get_recommendations)

        self.plot_low_price_high_rating_btn = wx.Button(self.panel, label='Rekomendasi Low Budget')
        self.hbox.Add(self.plot_low_price_high_rating_btn, flag=wx.EXPAND | wx.ALL, border=10)
        self.plot_low_price_high_rating_btn.Bind(wx.EVT_BUTTON, self.plot_low_price_high_rating)

        self.plot_high_price_high_rating_btn = wx.Button(self.panel, label='Rekomendasi High Budget')
        self.hbox.Add(self.plot_high_price_high_rating_btn, flag=wx.EXPAND | wx.ALL, border=10)
        self.plot_high_price_high_rating_btn.Bind(wx.EVT_BUTTON, self.plot_high_price_high_rating)

        self.evaluate_btn = wx.Button(self.panel, label='Evaluasi Model')
        self.hbox.Add(self.evaluate_btn, flag=wx.EXPAND | wx.ALL, border=10)
        self.evaluate_btn.Bind(wx.EVT_BUTTON, self.on_evaluate)

        self.clear_btn = wx.Button(self.panel, label='Hapus')
        self.hbox.Add(self.clear_btn, flag=wx.EXPAND | wx.ALL, border=10)
        self.clear_btn.Bind(wx.EVT_BUTTON, self.clear_output)

        self.vbox.Add(self.hbox, flag=wx.ALIGN_CENTER)

        self.result_text = wx.TextCtrl(self.panel, style=wx.TE_MULTILINE | wx.TE_READONLY)
        self.vbox.Add(self.result_text, proportion=1, flag=wx.EXPAND | wx.ALL, border=10)

        self.panel.SetSizer(self.vbox)
        self.panel.SetBackgroundColour(wx.Colour(40, 178, 170))

        self.df = None
        self.user_similarity_matrix = None


    #function dari input data
    def on_load(self, event):
        with wx.FileDialog(self, "Input Data", wildcard="CSV files (*.csv)|*.csv",
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return
            path = fileDialog.GetPath()
            try:
                self.df = pd.read_csv(path)
                self.preprocess_data()
                wx.MessageBox('Data berhasil dimuat!', 'Info', wx.OK | wx.ICON_INFORMATION)
            except Exception as e:
                wx.LogError(f"Tidak dapat memuat file '{path}'. Kesalahan: {e}")


    #function preprocesing data
    def preprocess_data(self):
        self.df['Harga'] = self.df['Harga'].replace('[Rp,.]', '', regex=True).astype(float)
        self.df.dropna(inplace=True)
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(self.df[['Rating', 'Harga']])
        df_scaled = pd.DataFrame(scaled_features, columns=['Scaled_Rating', 'Scaled_Harga'])
        self.user_similarity_matrix = cosine_similarity(df_scaled)


    #tampilan pesa error ketika data belum dimasukan /input
    def display_data(self, event):
        if self.df is not None:
            self.result_text.SetValue(self.df.to_string(index=False))
        else:
            wx.MessageBox('Muat data terlebih dahulu!', 'Info', wx.OK | wx.ICON_INFORMATION)


    #algoritma collaborative filltering
    def get_collaborative_filtering_recommendations(self, place_index, num_recommendations=5):
        if place_index >= len(self.user_similarity_matrix):
            return pd.DataFrame()  # Kembalikan DataFrame kosong jika terjadi kesalahan indeks

        place_similarity_scores = self.user_similarity_matrix[place_index]
        similar_places_indices = place_similarity_scores.argsort()[::-1][1:num_recommendations + 1]
        recommended_places = self.df.iloc[similar_places_indices]
        return recommended_places

    #get rekomendasi
    def get_recommendations(self, event):
        if self.df is not None and self.user_similarity_matrix is not None:
            try:
                recommendations = self.get_collaborative_filtering_recommendations(0, num_recommendations=10)
                if not recommendations.empty:
                    self.display_recommendations(recommendations)
                    self.plot_recommendations(recommendations)
            except Exception as e:
                wx.MessageBox(f"Terjadi kesalahan: {e}", 'Error', wx.OK | wx.ICON_ERROR)
        else:
            wx.MessageBox('Muat data terlebih dahulu!', 'Info', wx.OK | wx.ICON_INFORMATION)


    #tampilkan ahsil rekomendasi
    def display_recommendations(self, recommendations):
        # Hapus kolom 'Review' dari DataFrame recommendations jika ada
        if 'Review' in recommendations.columns:
            recommendations = recommendations.drop(columns=['Review'])

        col_widths = [max(len(str(value)) for value in recommendations[col].values) for col in recommendations.columns]
        col_widths = [max(width, len(col)) for width, col in zip(col_widths, recommendations.columns)]

        table_header = " | ".join(f"{col.ljust(width)}" for col, width in zip(recommendations.columns, col_widths))
        table_divider = "-+-".join('-' * width for width in col_widths)

        table_rows = "\n".join(
            " | ".join(f"{str(value).ljust(width)}" for value, width in zip(row, col_widths)) for row in
            recommendations.values)

        table_output = f"{table_header}\n{table_divider}\n{table_rows}"

        self.result_text.SetFont(wx.Font(12, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        self.result_text.SetValue(table_output)


    #visual rekomendasi
    def plot_recommendations(self, recommendations):
        plt.figure(figsize=(8, 6))
        bars = plt.barh(recommendations['Nama_Tempat'], recommendations['Rating'], color='green')
        plt.xlabel('Rating')
        plt.ylabel('Nama Tempat')
        plt.title('Rekomendasi Tempat (Top 10)')
        plt.gca().invert_yaxis()
        plt.subplots_adjust(left=0.3)
        for bar in bars:
            plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f'{bar.get_width():.2f}', va='center')
        plt.show()


    #tampilan rekomendasi harga rendah
    def plot_low_price_high_rating(self, event):
        if self.df is not None:
            low_price_high_rating = self.df[
                (self.df['Harga'] < self.df['Harga'].median()) & (self.df['Rating'] > self.df['Rating'].median())]
            low_price_high_rating_top10 = low_price_high_rating.head(10)  # Ambil 10 teratas
            plt.figure(figsize=(8, 6))  # Ukuran horizontal
            bars = plt.barh(low_price_high_rating_top10['Nama_Tempat'], low_price_high_rating_top10['Rating'], color='blue')
            plt.xlabel('Rating')
            plt.ylabel('Nama Tempat')
            plt.title('Low Budget (Top 10)')
            plt.gca().invert_yaxis()  # Balik urutan dari atas ke bawah
            plt.subplots_adjust(left=0.3)  # Sesuaikan margin kiri untuk menampilkan nama tempat
            for bar in bars:
                plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f'{bar.get_width():.2f}', va='center')
            plt.show()
        else:
            wx.MessageBox('Muat data terlebih dahulu!', 'Info', wx.OK | wx.ICON_INFORMATION)

    #tampilan rekomendasi harga tinggi
    def plot_high_price_high_rating(self, event):
        if self.df is not None:
            high_price_high_rating = self.df[
                (self.df['Harga'] > self.df['Harga'].median()) & (self.df['Rating'] > self.df['Rating'].median())]
            high_price_high_rating_top10 = high_price_high_rating.head(10)  # Ambil 10 teratas
            plt.figure(figsize=(8, 6))  # Ukuran horizontal
            bars = plt.barh(high_price_high_rating_top10['Nama_Tempat'], high_price_high_rating_top10['Rating'], color='red')
            plt.xlabel('Rating')
            plt.ylabel('Nama Tempat')
            plt.title('High Budget (Top 10)')
            plt.gca().invert_yaxis()  # Balik urutan dari atas ke bawah
            plt.subplots_adjust(left=0.3)  # Sesuaikan margin kiri untuk menampilkan nama tempat
            for bar in bars:
                plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f'{bar.get_width():.2f}', va='center')
            plt.show()
        else:
            wx.MessageBox('Muat data terlebih dahulu!', 'Info', wx.OK | wx.ICON_INFORMATION)

    def clear_output(self, event):
        self.result_text.Clear()


    #evaluasi model
    def evaluate_model(self):
        train_data, test_data = train_test_split(self.df, test_size=0.2, random_state=42)

        true_ratings = []
        predicted_ratings = []

        for index, row in test_data.iterrows():
            try:
                place_index = self.df.index[self.df['Nama_Tempat'] == row['Nama_Tempat']][0]
                predicted_places = self.get_collaborative_filtering_recommendations(place_index, 1)
                if not predicted_places.empty:
                    true_ratings.append(row['Rating'])
                    predicted_ratings.append(predicted_places.iloc[0]['Rating'])
            except IndexError:
                continue  # Lewati jika terjadi kesalahan indeks

        mae = mean_absolute_error(true_ratings, predicted_ratings)
        rmse = np.sqrt(mean_squared_error(true_ratings, predicted_ratings))  # Hitung RMSE dengan cara yang baru

        self.result_text.SetValue(f'Mean Absolute Error (MAE): {mae}\nRoot Mean Squared Error (RMSE): {rmse}')

    def on_evaluate(self, event):
        if self.df is not None and self.user_similarity_matrix is not None:
            self.evaluate_model()
        # else:
        #     wx.MessageBox('Muat data terlebih dahulu!', 'Info', wx.OK | wx.ICON_INFORMATION)


def main():
    app = wx.App(False)
    frame = RecommendationApp(None, title='Rekomendasi Tempat di Jakarta')
    frame.Show(True)
    app.MainLoop()


if __name__ == "__main__":
    main()

#selesai deng
