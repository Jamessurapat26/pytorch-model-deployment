from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image

app = Flask(__name__)

# Load model for CPU
model = torch.load('mobilenetv3_large_100_checkpoint_fold0.pt', map_location=torch.device('cpu'))
model = model.half()  # Convert the model to half precision
model.eval()

# Assuming you have a list of class names
class_names = ["AD", "Control", "PD"]  # Replace with your actual class names

# Define the transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250]))
])


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image']

    try:
        img = Image.open(image).convert('RGB')
    except IOError:
        return jsonify({'error': 'Invalid image'}), 400

    try:
        img_tensor = transform(img).unsqueeze(0)
    except:
        return jsonify({'error': 'Error during image preprocessing'}), 500

    # Make prediction
    try:
        with torch.no_grad():
            output = model(img_tensor)

        # Get the predicted class index and name
        pred_idx = output.argmax().item()
        pred_class = class_names[pred_idx]
        confidence = float(torch.softmax(output, dim=1).max().item())

        return jsonify({
            'prediction': pred_class,
            'confidence': confidence,
            'class_index': pred_idx
        })
    except Exception as e:
        return jsonify({'error': f'Error during prediction: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True)