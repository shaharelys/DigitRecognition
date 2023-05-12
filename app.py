from flask import Flask, request, render_template
from PIL import Image
from torchvision.transforms import ToTensor
from model import Net
from config import MODEL_PATH
import torch
from werkzeug.utils import secure_filename
import os
from PIL import ImageOps

app = Flask(__name__)

# Load the trained model
model = Net()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        image = Image.open(file).convert('L')
        image = ImageOps.invert(image)  # Invert colors
        image = image.resize((28, 28))
        image = ToTensor()(image).unsqueeze(0)
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        return render_template('result.html', prediction=int(predicted))
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, port=8000)
