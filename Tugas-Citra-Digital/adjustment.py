import cv2              # Mengimpor library OpenCV untuk manipulasi gambar
import numpy as np      # Mengimpor NumPy untuk membuat array dan operasi matematika

# =========================
# 1. Load Gambar
# =========================
img = cv2.imread("citra.jpg")        # Membaca gambar bernama "citra.jpg" dari folder proyek

# =========================
# 2. Color Correction (CLAHE)
# =========================
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)   # Mengubah format warna gambar dari BGR ke LAB
l, a, b = cv2.split(lab)                     # Memecah gambar LAB menjadi channel L, A, dan B

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))   # Membuat objek CLAHE untuk meningkatkan kontras
cl = clahe.apply(l)                                           # Menerapkan CLAHE hanya pada channel L (lightness)

lab_clahe = cv2.merge((cl, a, b))              # Menggabungkan kembali channel L, A, dan B yang sudah dikoreksi
corrected_img = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)   # Mengubah gambar kembali ke format BGR

# =========================
# 3. Sharpening (Kernel)
# =========================
kernel = np.array([[0, -1, 0],       # Kernel/sharpening filter
                   [-1, 5, -1],      # Nilai positif di tengah untuk mempertajam tepi gambar
                   [0, -1, 0]])
sharpened = cv2.filter2D(corrected_img, -1, kernel)   # Menerapkan filter sharpening ke gambar

# =========================
# 4. Brightness Adjustment
# =========================
brightness = cv2.convertScaleAbs(sharpened, alpha=1.2, beta=20)
# alpha = meningkatkan kontras (1.2 berarti 20% lebih kuat)
# beta  = menambah kecerahan sebesar 20 piksel
# Hasilnya gambar jadi lebih terang dan lebih jelas

# =========================
# 5. Save Output
# =========================
cv2.imwrite("hasil_edit.jpg", brightness)  # Menyimpan hasil gambar akhir dengan nama "hasil_edit.jpg"

# =========================
# 6. Preview (opsional)
# =========================
cv2.imshow("Hasil Edit", brightness)   # Menampilkan hasil edit dalam jendela tampilan
cv2.waitKey(0)                         # Menunggu sampai ada tombol keyboard ditekan
cv2.destroyAllWindows()                # Menutup semua jendela OpenCV
