from flask import Flask, render_template, request, Response, redirect, url_for, flash
import joblib
import os
import numpy as np
import io
import base64
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.patches import Arc, FancyArrow
app = Flask(__name__)
app.secret_key = 'nguyenduongtuan'
@app.route("/")
def index():
    return render_template('home.html')

@app.route("/result", methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        try:
            # Nhận dữ liệu từ form
            gender = int(request.form['gender'])
            age = int(request.form['age'])
            hypertension = int(request.form['hypertension'])
            heart_disease = int(request.form['heart_disease'])
            ever_married = int(request.form['ever_married'])
            work_type = int(request.form['work_type'])
            Residence_type = int(request.form['Residence_type'])
            avg_glucose_level = float(request.form['avg_glucose_level'])
            bmi = float(request.form['bmi'])
            smoking_status = int(request.form['smoking_status'])

            # Tạo mảng numpy từ dữ liệu
            x = np.array([gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type,
                          avg_glucose_level, bmi, smoking_status]).reshape(1, -1)

            # Đường dẫn đến file mô hình và scaler
            scaler_path = os.path.join('scaler.pkl')
            model_path = os.path.join('RandomForest_model.pkl')

            # Tải mô hình và scaler
            scaler = joblib.load(scaler_path)
            model = joblib.load(model_path)

            # Chuẩn hóa dữ liệu đầu vào
            x_scaled = scaler.transform(x)

            # Dự đoán và tính toán xác suất
            probabilities = model.predict_proba(x_scaled)[0]
            stroke_probability = probabilities[1]  # Xác suất của lớp 1 (có nguy cơ đột quỵ)
            no_stroke_probability = probabilities[0]  # Xác suất của lớp 0 (không có nguy cơ đột quỵ)

            # Xác định kết quả dự đoán
            if stroke_probability >= 0.5:
                percentage = round(stroke_probability * 100, 2)
                result = "có nguy cơ đột quỵ"
            else:
                percentage = round(no_stroke_probability * 100, 2)
                result = "không có nguy cơ đột quỵ"

            # Tạo hình cung
            fig, ax = plt.subplots()

            # Tính toán góc theo tỷ lệ phần trăm (trong độ)
            angle = 180 * (percentage / 100)

            # Vẽ cung màu trắng (khoảng không) từ 0 đến 180 độ
            arc_white = Arc((0, 0), 2, 2, angle=0, theta1=0, theta2=180, color='lightgrey', lw=10)
            ax.add_patch(arc_white)

            # Vẽ cung màu đỏ từ 0 đến tỷ lệ phần trăm
            arc_color = Arc((0, 0), 2, 2, angle=0, theta1=180 - angle, theta2=180, color='red', lw=10)
            ax.add_patch(arc_color)

            # Vẽ kim chỉ báo
            arrow = FancyArrow(0, 0, np.cos(np.radians(180 - angle)) * 0.8, np.sin(np.radians(180 - angle)) * 0.8,
                            color='black', width=0.05, length_includes_head=True)
            ax.add_patch(arrow)

            # Hiển thị các mốc chỉ số
            ticks = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            for tick in ticks:
                angle_tick = 180 * (tick / 100)  # Tính góc tương ứng với mốc
                x = 0.9 * np.cos(np.radians(180 - angle_tick))  # Tính tọa độ x
                y = 0.9 * np.sin(np.radians(180 - angle_tick))  # Tính tọa độ y
                ax.text(x, y, str(tick), horizontalalignment='center', fontsize=10)

            # Hiển thị tỷ lệ phần trăm
            ax.text(0, -0.25, f'{percentage}%', horizontalalignment='center', fontsize=16, fontweight='bold')

            # Thiết lập trục và hiển thị hình ảnh
            ax.set_aspect('equal')
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-0.2, 1.2)
            ax.axis('off')

            # Save to a bytes buffer
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            buf.close()

            # Trả về trang với kết quả
            return render_template('home.html', percentage=percentage, result=result, gauge_chart=img_base64)

        except ValueError:
            # Nếu có bất kỳ giá trị nào không thể chuyển đổi thành số hoặc bị thiếu, thông báo lỗi
            flash("Vui lòng điền đầy đủ thông tin vào tất cả các trường trước khi nhấn Submit.")
            return redirect(url_for('index'))

    # Nếu là yêu cầu GET, trả về trang chủ
    return render_template('home.html')
if __name__ == "__main__":
    app.run(debug=True, port=7384)




