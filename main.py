from fastapi import FastAPI, UploadFile, File, HTTPException
import io
import uvicorn
import torch
from torchvision import transforms
import torch.nn as nn
from PIL import  Image
import streamlit as st

classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
class CheckDataset(nn.Module):
  def __init__(self):
    super().__init__()

    self.first = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )

    self.second = nn.Sequential(
        nn.Flatten(),
        nn.Linear(64 * 32 * 32, 128),
        nn.ReLU(),
        nn.Linear(128, 6)
    )

  def forward(self, x):
   x = self.first(x)
   x = self.second(x)
   return x

transform_data = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(),
    transforms.ToTensor()
])

check_datasets = FastAPI()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CheckDataset()
model.load_state_dict(torch.load('model_datasets.pth', map_location=device))
model.to(device)
model.eval()

st.title('Garbage Datasets')
st.write('Загрузите изображение модель попробует распознать её')

image= st.file_uploader('Выберити изброжение', type=['png', 'jpg', 'jpeg ', 'svg'])

if not image:
    st.info('Загружити изоброжение')
else:
    st.image(image, caption='Загружоное избражение')

    if st.button('Опредилит изброжение'):
        try:
            image_data = image.read()

            if not image_data:
                raise HTTPException(status_code=400, detail='Файл кошулган жок')
            img = Image.open(io.BytesIO(image_data))
            img_tensor = transform_data(img).unsqueeze(0).to(device)

            with torch.no_grad():
                y_pred = model(img_tensor)
                pred = y_pred.argmax(dim=1).item()
                pred_name = classes[pred]
            st.success(f'Индекс: {pred}, изоброжение: {pred_name}')
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

# @check_datasets.post('/predict/')
# async def check_image(file: UploadFile = File(...)):
#     try:
#         image_data = await file.read()
#
#         if not image_data:
#             raise  HTTPException(status_code=400, detail='Файл кошулган жок')
#
#         img = Image.open(io.BytesIO(image_data))
#         img_tensor = transform_data(img).unsqueeze(0).to(device)
#
#         with torch.no_grad():
#             y_pred = model(img_tensor)
#             pred = y_pred.argmax(dim=1).item()
#
#         return {'Answer': pred, 'label': classes[pred]}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
#
#
# if __name__ == '__main__':
#     uvicorn.run(check_datasets, host='127.0.0.1', port=8000)