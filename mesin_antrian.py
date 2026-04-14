import tkinter as tk
from tkinter import simpledialog
import win32print
import win32ui
from datetime import datetime
import os
from PIL import Image, ImageTk, ImageDraw, ImageFont


base_dir = os.path.dirname(os.path.abspath(__file__))
bg_path = os.path.join(base_dir, "New folder", "bg.png")
config_path = os.path.join(base_dir, "config.txt")

logo_path = r"C:\Users\IT\PycharmProjects\PythonProject\.venv\Scripts\New folder\logo.png"
COLOR_MAIN = "#005baa"
COLOR_HOVER = "#003f7d"
TEXT_COLOR = "white"
def load_judul():
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return f.read().strip()
    return "MESIN ANTRIAN BRI"

def save_judul(text):
    with open(config_path, "w") as f:
        f.write(text)

judul = load_judul()

teller = 1
cs = 1

def load_logo_escpos(path):
    try:
        img = Image.open(path).convert("L")
        img = img.resize((384, int(img.height * 384 / img.width)))
        img = img.point(lambda x: 0 if x < 128 else 255, '1')

        width, height = img.size
        pixels = img.load()

        data = b''
        for y in range(0, height, 24):
            data += b'\x1b\x2a\x21' + bytes([width % 256, width // 256])
            for x in range(width):
                byte = 0
                for bit in range(8):
                    if y + bit < height:
                        if pixels[x, y + bit] == 0:
                            byte |= (1 << (7 - bit))
                data += bytes([byte])
            data += b'\n'
        return data
    except:
        return b''
def print_escpos(layanan, nomor):
    now = datetime.now()
    waktu = now.strftime("%A, %d %B %Y %H:%M:%S")

    tiket = f"""
{judul}
--------------------------------
{layanan}

        {nomor}

MENUNGGU : -

{waktu}

Terima Kasih Atas Kunjungan Anda
"""

    printer = win32print.GetDefaultPrinter()
    handle = win32print.OpenPrinter(printer)
    win32print.StartDocPrinter(handle, 1, ("Tiket", None, "RAW"))
    win32print.StartPagePrinter(handle)
    logo_bytes = load_logo_escpos(logo_path)
    if logo_bytes:
        win32print.WritePrinter(handle, logo_bytes)
    win32print.WritePrinter(handle, tiket.encode("utf-8"))
    win32print.WritePrinter(handle, b'\n\n\n\x1d\x56\x00')
    win32print.EndPagePrinter(handle)
    win32print.EndDocPrinter(handle)
    win32print.ClosePrinter(handle)

def print_windows(layanan, nomor):
    now = datetime.now()
    waktu = now.strftime("%A, %d %B %Y %H:%M:%S")

    width = 384
    height = 600

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    try:
        font_big = ImageFont.truetype("arial.ttf", 60)
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font_big = None
        font = None

    y = 10

    # LOGO
    if os.path.exists(logo_path):
        logo = Image.open(logo_path)
        logo = logo.resize((200, int(logo.height * 200 / logo.width)))
        img.paste(logo, ((width - logo.width)//2, y))
        y += logo.height + 10

    draw.text((width//2, y), judul, fill="black", anchor="mm", font=font)
    y += 40

    draw.text((width//2, y), layanan, fill="black", anchor="mm", font=font)
    y += 50

    draw.text((width//2, y), nomor, fill="black", anchor="mm", font=font_big)
    y += 100

    draw.text((width//2, y), "MENUNGGU : -", fill="black", anchor="mm", font=font)
    y += 40

    draw.text((width//2, y), waktu, fill="black", anchor="mm", font=font)

    printer = win32print.GetDefaultPrinter()
    hDC = win32ui.CreateDC()
    hDC.CreatePrinterDC(printer)

    hDC.StartDoc("Tiket")
    hDC.StartPage()

    dib = ImageWin.Dib(img)
    dib.draw(hDC.GetHandleOutput(), (0, 0, width, height))

    hDC.EndPage()
    hDC.EndDoc()
    hDC.DeleteDC()

def print_ticket(layanan, nomor):
    try:
        print_escpos(layanan, nomor)
    except Exception as e:
        print("ESC/POS gagal, pindah ke printer biasa:", e)
        try:
            print_windows(layanan, nomor)
        except Exception as e2:
            print("Print Windows gagal:", e2)


app = tk.Tk()
app.title("MESIN ANTRIAN")
app.attributes("-fullscreen", True)

canvas = tk.Canvas(app, highlightthickness=0)
canvas.pack(fill="both", expand=True)

bg_photo = None
bg_id = None

judul_text = canvas.create_text(0, 0, text=judul,
                                font=("Arial", 32, "bold"),
                                fill="white",
                                anchor="center")

nomor_text = canvas.create_text(0, 0, text="",
                                font=("Arial", 100, "bold"),
                                fill="white",
                                anchor="center")

waktu_text = canvas.create_text(0, 0,
                                font=("Arial", 18, "bold"),
                                fill="white",
                                anchor="center")

class RoundedButton:
    def __init__(self, canvas, relx, rely, relwidth, relheight, text, command):
        self.canvas = canvas
        self.relx = relx
        self.rely = rely
        self.relwidth = relwidth
        self.relheight = relheight
        self.text = text
        self.command = command
        self.frame = None
        self.button = None
        self.txt = None

    def round_rectangle(self, x1, y1, x2, y2, r=25, **kwargs):
        points = [
            x1+r, y1, x2-r, y1, x2, y1, x2, y1+r, x2, y2-r, x2, y2,
            x2-r, y2, x1+r, y2, x1, y2, x1, y2-r, x1, y1+r, x1, y1
        ]
        return self.canvas.create_polygon(points, **kwargs, smooth=True)

    def draw(self, width, height):
        if self.frame: self.canvas.delete(self.frame)
        if self.button: self.canvas.delete(self.button)
        if self.txt: self.canvas.delete(self.txt)

        x1 = width*self.relx - (width*self.relwidth)/2
        y1 = height*self.rely - (height*self.relheight)/2
        x2 = width*self.relx + (width*self.relwidth)/2
        y2 = height*self.rely + (height*self.relheight)/2

        self.frame = self.round_rectangle(x1-5, y1-5, x2+5, y2+5, r=25, fill="white")
        self.button = self.round_rectangle(x1, y1, x2, y2, r=20, fill=COLOR_MAIN)

        font_size = max(12, int((y2-y1)/3))
        self.txt = self.canvas.create_text((x1+x2)/2, (y1+y2)/2,
                                           text=self.text,
                                           font=("Arial", font_size, "bold"),
                                           fill=TEXT_COLOR)

        self.canvas.tag_bind(self.button, "<Button-1>", lambda e: self.command())
        self.canvas.tag_bind(self.txt, "<Button-1>", lambda e: self.command())

        self.canvas.tag_bind(self.button, "<Enter>",
                             lambda e: self.canvas.itemconfig(self.button, fill=COLOR_HOVER))
        self.canvas.tag_bind(self.button, "<Leave>",
                             lambda e: self.canvas.itemconfig(self.button, fill=COLOR_MAIN))

        self.canvas.tag_bind(self.txt, "<Enter>",
                             lambda e: self.canvas.itemconfig(self.button, fill=COLOR_HOVER))
        self.canvas.tag_bind(self.txt, "<Leave>",
                             lambda e: self.canvas.itemconfig(self.button, fill=COLOR_MAIN))

def ambil_teller():
    global teller
    nomor = f"A{teller:03}"
    canvas.itemconfig(nomor_text, text=nomor)
    print_ticket("TELLER", nomor)
    teller += 1

def ambil_cs():
    global cs
    nomor = f"B{cs:03}"
    canvas.itemconfig(nomor_text, text=nomor)
    print_ticket("CUSTOMER SERVICE", nomor)
    cs += 1

def reset_antrian():
    global teller, cs
    teller = 1
    cs = 1
    canvas.itemconfig(nomor_text, text="Reset")
    app.after(1000, lambda: canvas.itemconfig(nomor_text, text=""))

def ubah_judul():
    global judul
    new_judul = simpledialog.askstring("Ubah Judul", "Masukkan judul baru:")
    if new_judul:
        judul = new_judul
        canvas.itemconfig(judul_text, text=judul)

        width = canvas.winfo_width()
        height = canvas.winfo_height()
        canvas.coords(judul_text, width/2, height*0.08)

        save_judul(judul)

menu = tk.Menu(app, tearoff=0)
menu.add_command(label="Reset Antrian", command=reset_antrian)
menu.add_command(label="Ubah Judul", command=ubah_judul)
menu.add_separator()
menu.add_command(label="Exit", command=app.destroy)

def show_menu(event):
    menu.tk_popup(event.x_root, event.y_root)

app.bind("<Button-3>", show_menu)

def create_buttons():
    app.teller_btn = RoundedButton(canvas, 0.5, 0.53, 0.35, 0.12, "TELLER", ambil_teller)
    app.cs_btn = RoundedButton(canvas, 0.5, 0.72, 0.35, 0.12, "CUSTOMER SERVICE", ambil_cs)

create_buttons()

def resize_all(event):
    width, height = event.width, event.height

    global bg_photo, bg_id
    if os.path.exists(bg_path):
        img = Image.open(bg_path)
        img = img.resize((width, height), Image.Resampling.LANCZOS)
        bg_photo = ImageTk.PhotoImage(img)

        if bg_id is None:
            bg_id = canvas.create_image(0, 0, image=bg_photo, anchor="nw")
        else:
            canvas.itemconfig(bg_id, image=bg_photo)

        canvas.tag_lower(bg_id)

    canvas.coords(judul_text, width/2, height*0.08)
    canvas.coords(nomor_text, width/2, height*0.30)
    canvas.coords(waktu_text, width/2, height*0.92)

    app.teller_btn.draw(width, height)
    app.cs_btn.draw(width, height)

canvas.bind("<Configure>", resize_all)

def update_waktu():
    hari = ["Senin","Selasa","Rabu","Kamis","Jumat","Sabtu","Minggu"]
    now = datetime.now()
    hari_ini = hari[now.weekday()]
    teks = f"{hari_ini}, {now.strftime('%d-%m-%Y | %H:%M:%S')}"
    canvas.itemconfig(waktu_text, text=teks)
    app.after(1000, update_waktu)

update_waktu()

app.bind("<Escape>", lambda e: app.destroy())

app.mainloop()