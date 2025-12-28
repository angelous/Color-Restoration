# Image Colorization menggunakan ResNet18 U-Net
Projek ini berfokus pada restorasi pada citra grayscale secara otomatis menggunakan Deep Learning. 
Model yang digunakan berbasis arsitektur **U-Net** dengan **ResNet18 pretrained** sebagai encoder, yang di-fine-tune untuk memprediksi informasi warna pada ruang warna LAB.

## Project Structure
- app.py                           : Streamlit deployment (Render)
- computer_vision_script.ipynb     : Training & evaluation
- model_colorization_tuned (2).pth : Trained ResNet18 U-Net model
- requirements.txt                 : Dependencies

## Deployment
Model dideploy menggunakan **Streamlit**. Aplikasi ini memungkinkan pengguna untuk melakukan restorasi warna pada citra grayscale melalui web tanpa melakukan proses training ulang.

### Alur Sistem Deployment
1. Pengguna mengupload citra grayscale melalui web
2. Sistem akan melakukan preprocessing, meliputi :
   - Resize ke ukuran 256 × 256 pixel
   - Konversi ke LAB Color
   - Extract L channel sebagai input model
4. L channel akan diproses oleh model untuk memprediksi AB channel
5. L channel dan AB channel digabungkan kembali lalu dikonversi menjadi RGB
6. Hasil akan ditampilkan kepada pengguna dan dapat diunduh dalam format PNG

### Link Deployment
https://color-restoration.onrender.com/
<img width="1919" height="1027" alt="image" src="https://github.com/user-attachments/assets/b3c81cd8-ec68-4502-ac7d-27fd89c3be83" />


## 🛠️ Cara Menjalankan
```bash
pip install -r requirements.txt
streamlit run app.py
