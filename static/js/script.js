function predict() {
    const fileInput = document.getElementById('imageUpload');
    if (!fileInput.files.length) {
        alert('Vui lòng chọn một ảnh.');
        return;
    }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert(data.error);
            return;
        }

        // Hiển thị kết quả dự đoán
        document.getElementById('prediction').innerHTML = 
            `Lớp dự đoán: ${data.pred_label} (${data.pred_confidence.toFixed(2)}%)<br>` +
            `Xác suất:<br>` +
            Object.entries(data.probabilities).map(([cls, prob]) => `${cls}: ${prob.toFixed(2)}%`).join('<br>');

        // Hiển thị ảnh Grad-CAM
        const gradcamImage = document.getElementById('gradcamImage');
        gradcamImage.src = '../' + data.gradcam_image + '?' + new Date().getTime();
        gradcamImage.style.display = 'block';

        // Hiển thị biểu đồ xác suất
        const probPlot = document.getElementById('probPlot');
        probPlot.src = 'data:image/png;base64,' + data.plot_base64;
        probPlot.style.display = 'block';
    })
    .catch(error => {
        console.error('Lỗi:', error);
        alert('Đã xảy ra lỗi trong quá trình dự đoán.');
    });
}