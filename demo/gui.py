import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
from tqdm import tqdm
from runcontrol0921 import autoencoder
import torch
import numpy as np
import matplotlib.pyplot as plt

class VideoProcessor:
    def __init__(self, root):
        self.root = root
        self.root.title("视频处理工具")
        self.root.geometry("2000x1000")

        self.upload_button = tk.Button(root, text="上传视频", command=self.upload_video)
        self.upload_button.pack()

        self.progress_bar = ttk.Progressbar(root, orient="horizontal", mode="determinate")
        self.progress_bar.pack(fill=tk.X, padx=10, pady=10)

        self.canvas = tk.Canvas(root, width=896, height=448)
        self.canvas.pack()

        self.rect = None
        self.start_x = None
        self.start_y = None
        self.crop_coords = None
        self.frames = []

        self.confirm_button = tk.Button(root, text="确认框选", command=self.confirm_selection, state=tk.DISABLED)
        self.confirm_button.pack()

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        self.model=autoencoder.load_from_checkpoint(r'D:\PIV_code\GUI\model0921.ckpt')

    def upload_video(self):
        video_path = filedialog.askopenfilename(title="选择视频", filetypes=[("视频文件", "*.mp4;*.avi")])
        if video_path:
            self.process_video(video_path)

    def process_video(self, videofile):
        cap = cv2.VideoCapture(videofile)
        if not cap.isOpened():
            messagebox.showerror("错误", "无法打开视频文件")
            return

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames: {total}")

        self.progress_bar["maximum"] = total
        self.progress_bar["value"] = 0

        print("读取第一帧...")
        flag, frame = cap.read()
        if not flag:
            messagebox.showerror("错误", "无法读取第一帧")
            return

        # 直接展示未处理的第一帧
        self.display_first_frame(frame)

        # 读取所有帧并存储
        for i in tqdm(range(total)):
            flag, frame = cap.read()
            if not flag:
                print(f"无法读取帧 {i}")
                break
            # 转为灰度图像
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.frames.append(gray_frame)

            self.progress_bar["value"] += 1
            self.root.update_idletasks()

    def display_first_frame(self, frame):
        img = Image.fromarray(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (896, 448)))
        self.tk_img = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

    def on_button_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='red')
        self.confirm_button.config(state=tk.NORMAL)

    def on_button_release(self, event):
        self.end_x = event.x
        self.end_y = event.y
        self.canvas.coords(self.rect, self.start_x, self.start_y, self.end_x, self.end_y)

    def confirm_selection(self):
        self.crop_coords = (min(self.start_x, self.end_x), min(self.start_y, self.end_y),
                            max(self.start_x, self.end_x), max(self.start_y, self.end_y))
        self.extract_background()
        self.confirm_button.config(state=tk.DISABLED)

    def extract_background(self):
        print("根据框选区域计算背景模型...")
        x1, y1, x2, y2 = self.crop_coords
        cropped_frames = []

        # 裁剪并调整大小
        for frame in self.frames:
            cropped_frame = frame[:, 650:] # 裁剪这一部分未传入参数
            resized_frame = cv2.resize(cropped_frame, (896, 448))  # 调整大小
            cropped_frames.append(resized_frame)

        del self.frames

        background_model = np.median(np.array(cropped_frames), axis=0).astype(np.uint8)

        # 后续前景提取逻辑
        processed_images = []
        for resized_frame in cropped_frames:
            foreground = cv2.absdiff(resized_frame, background_model)
            _, binary_frame = cv2.threshold(foreground, 55, 255, cv2.THRESH_BINARY)
            processed_images.append(binary_frame)

        del cropped_frames

        self.dilate_images(processed_images)

    def dilate_images(self, processed_images):
        print("执行膨胀操作...")
        kernel = np.ones((2, 2), np.uint8)
        dilated_images = []

        for image in processed_images:
            dilated_image = cv2.dilate(image, kernel, iterations=1)
            dilated_images.append(dilated_image)

        del processed_images

        print("膨胀操作完成！")

        # 展示膨胀后的第一帧
        self.display_dilated_first_frame(dilated_images[0])


        self.predict(np.array(dilated_images))

    def display_dilated_first_frame(self, frame):
        img = Image.fromarray(frame)
        self.tk_img = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

    def predict(self, frame):
        new_size = (frame.shape[0] // 4) * 4
        frame = frame[:new_size]  # 截取数组
        frame = torch.from_numpy(frame.reshape(-1, 4, 448, 896)).float()

        batch_size = 32
        results = []

        for i in range(0, frame.shape[0], batch_size):
            batch = frame[i:i + batch_size]
            batch_results = self.model(batch).cpu().detach().numpy().reshape(-1, 1, 448, 896)
            results.append(batch_results)

        # 将结果合并为一个数组
        results = np.concatenate(results, axis=0)
        print(results.shape)

        del frame

        # 直接生成视频
        self.create_video_from_results(results)

    def create_video_from_results(self, results):

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4编码
        out = cv2.VideoWriter('output_video.mp4', fourcc, 30.0, (896, 448))  # 输出视频的参数

        for i in range(len(results)):
            img = results[i]  # 选择当前结果的第一个通道
            img = img.squeeze(0)  # 去掉通道维度，变为 (448, 896)

            img = (img * 255).astype(np.float32)  # 将图像值放大到 0-255

            img_normalized = (img - img.min()) / (img.max() - img.min())  # 归一化到 0-1 范围
            img_normalized = (img_normalized * 255).astype(np.uint8)  # 转换为 uint8 格式

            # 应用色彩映射
            img_colored = plt.get_cmap('RdBu_r')(img_normalized)  # 使用 RdBu_r 颜色映射
            img_colored = (img_colored[:, :, :3] * 255).astype(np.uint8)  # 转换为 uint8 格式，去掉 alpha 通道

            out.write(img_colored)  # 写入视频文件

        out.release()  # 释放视频写入对象
        print("视频已生成：output_video.mp4")

    def save_snapshots(self, x):
        fig,ax=plt.subplots(1,1)
        pcm=ax.imshow(x,aspect='equal', cmap='RdBu_r')
        ax.axis('off')  # 关闭坐标轴
        fig.tight_layout()
        fig.show()
    #     

    # def create_video_from_results(self, results):
    #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4编码
    #     out = cv2.VideoWriter('output_video.mp4', fourcc, 30.0, (896, 448))  # 输出视频的参数

    #     for i in range(len(results)):
    #         # 读取快照图像
    #         img = cv2.imread(f"snapshots_{i}.png")
            
    #         if img is not None:
    #             out.write(img)  # 写入视频文件
    #         else:
    #             print(f"Warning: Could not read image snapshots_{i}.png")

    #     out.release()  # 释放视频写入对象
    #     print("视频已生成：output_video.mp4")



if __name__ == "__main__":
    root = tk.Tk()
    app = VideoProcessor(root)
    root.mainloop()
