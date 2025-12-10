import torch
import cv2
import numpy as np
import time

from torchvision import transforms
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weigths = torch.load('yolov7-w6-pose.pt')
model = weigths['model']
model = model.half().to(device)
_ = model.eval()

cap = cv2.VideoCapture(0)
if (cap.isOpened() == False):
    print('open failed.')

# 分辨率
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# 图片缩放
vid_write_image = letterbox(cap.read()[1], (frame_width), stride=64, auto=True)[0]
resize_height, resize_width = vid_write_image.shape[:2]

save_name = f"{'test-30.mp4' .split('/')[-1].split('.')[0]}"

# 保存结果视频
out = cv2.VideoWriter("test-100.mp4",
                      cv2.VideoWriter_fourcc(*'mp4v'), 30,
                      (resize_width, resize_height))

# frame_count = 0
# total_fps = 0

while(cap.isOpened):
    ret, frame = cap.read()
    if ret:
        orig_image = frame
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        image = letterbox(image, (frame_width), stride=64, auto=True)[0]
        image_ = image.copy()
        image = transforms.ToTensor()(image)
        image = torch.tensor(np.array([image.numpy()]))
        image = image.to(device)
        image = image.half()

        start_time = time.time()
        with torch.no_grad():
            output, _ = model(image)
        end_time = time.time()

        # 计算fps
        # fps = 1 / (end_time - start_time)
        # total_fps += fps
        # frame_count += 1

        output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
        output = output_to_keypoint(output)
        nimg = image[0].permute(1, 2, 0) * 255
        nimg = nimg.cpu().numpy().astype(np.uint8)
        nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
        for idx in range(output.shape[0]):
            plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)

        # 显示fps
        # cv2.putText(nimg, f"{fps:.3f} FPS", (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
        #             1, (0, 255, 0), 2)

        # 显示结果并保存
        cv2.imshow('image', nimg)
        out.write(nimg)

        # 按q退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# 资源释放
cap.release()
cv2.destroyAllWindows()

# 计算平均fps
# avg_fps = total_fps / frame_count
# print(f"Average FPS: {avg_fps:.3f}")