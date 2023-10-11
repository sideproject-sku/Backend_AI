import torch
from torch.autograd import Variable
import torchvision.transforms as standard_transforms
import numpy as np
from models.CC import CrowdCounter
from config import cfg
from PIL import Image
import cv2

torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True

mean_std = ([0.452016860247, 0.447249650955, 0.431981861591],[0.23242045939, 0.224925786257, 0.221840232611])

img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])

model_path = 'best.pth'
file_path = "test/"                                        

def main(camera_ID):
    net = CrowdCounter(cfg.GPU_ID,cfg.NET)
    net.load_state_dict(torch.load(model_path))
    net.cuda()
    net.eval()

    cam  = cv2.VideoCapture(file_path + camera_ID+'.mp4')

    if not cam.isOpened():
        print("disconnected")
        exit()

    alpha = 0.5

    peoples = []

    inc = False

    while True:

        status, img = cam.read()
        
        if status:
            im0 = img_transform(img)

            with torch.no_grad():
                im0 = Variable(im0[None,:,:,:]).cuda()
                pred_map = net.test_forward(im0)

            pred_map = pred_map.cpu().data.numpy()[0,0,:,:]

            pred = np.sum(pred_map)/100.0
            pred_map = pred_map/np.max(pred_map+1e-20)
            peoples.append(int(pred))
            pred_map = np.repeat(pred_map[:,:,np.newaxis], 3, -1)
            pred_map = (pred_map*255.0).astype(np.uint8)
            pred_map = cv2.applyColorMap(pred_map, cv2.COLORMAP_JET)

            res = cv2.addWeighted(img, 1 - alpha, pred_map, alpha, 0)


            cv2.putText(res,'counting : ' + str(int(pred)), (50,50), cv2.FONT_ITALIC, 1, (0,0,255), 2)

            previous_number = None


            for number in peoples:
                if previous_number is None or number > previous_number:
                    inc = True
                else:
                    inc = False
                previous_number = number

        

            if inc:
                top_left = (10, 10)  # (x, y) 좌표
                bottom_right = (res.shape[1] - 10, res.shape[0] - 10)
                cv2.rectangle(res, top_left, bottom_right, (0, 0, 255), 5) #경고표시

            status, buffer = cv2.imencode('.jpg',res)
            res = buffer.tobytes()
            yield (b'--frame\r\n'
                     b'Content-Type: image/jpeg\r\n\r\n' + res + b'\r\n')
        else:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    





