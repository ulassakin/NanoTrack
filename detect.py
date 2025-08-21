from multiprocessing import Process, Manager, freeze_support
from datetime import datetime as date
from loguru import logger

from glob import glob

import torch.cuda
import argparse
import cv2
import os
import numpy as np

from edgeyolo.detect import Detector, TRTDetector, draw
from copy import deepcopy
import json
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser("EdgeYOLO Detect parser")

    parser.add_argument("-w", "--weights", type=str, default="edgeyolo_coco.pth", help="weight file")
    parser.add_argument("-c", "--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("-n", "--nms-thres", type=float, default=0.55, help="nms threshold")

    parser.add_argument("--mp", action="store_true", help="use multi-process to accelerate total speed")

    parser.add_argument("--fp16", action="store_true", help="fp16")
    parser.add_argument("--no-fuse", action="store_true", help="do not fuse model")
    parser.add_argument("--input-size", type=int, nargs="+", default=[640, 640], help="input size: [height, width]")
    parser.add_argument("-s", "--source", type=str, default=None, help="video source/image dir/rosbag")
    parser.add_argument("--trt", action="store_true", help="is trt model")
    parser.add_argument("--legacy", action="store_true", help="if img /= 255 while training, add this command.")
    parser.add_argument("--use-decoder", action="store_true", help="support original yolox model v0.2.0")
    parser.add_argument("--batch", type=int, default=1, help="batch size")
    parser.add_argument("--no-label", action="store_true", help="do not draw label")
    parser.add_argument("--save-dir", type=str, default="./output/detect/imgs/", help="image result save dir")
    parser.add_argument("--fps", type=int, default=99999, help="max fps")

    parser.add_argument("--topic", type=str, default=None, help="ros or rosbag image topic")

    return parser.parse_args()


class ROSBase:

    def __init__(self):
        import numpy as np
        self.np = np

    def _compressedImage2CVMat(self, msg):
        return cv2.imdecode(self.np.frombuffer(msg.data, dtype=self.np.uint8), cv2.IMREAD_COLOR)


    def _Image2CVMat(self, msg):
        return self.np.reshape(self.np.fromstring(msg.data, dtype="bgr8"), [msg.height, msg.width, 3])
    


class ROSBagCapture(ROSBase):

    def __init__(self, bagfile: str, topic: str):
        super(ROSBagCapture, self).__init__()
        from rosbag import Bag
        
        self.bag = Bag(bagfile)
        self.len = self.bag.get_message_count(topic)
        self.iter = self.bag.read_messages(topic)
        self.idx = 0
        self.first = True
        self.compressed = True

    def isOpened(self):
        return self.idx < self.len - 1

    def read(self):
        if self.isOpened():
            topic, msg, timestamp = next(self.iter)
            self.idx += 1
            if self.first:
                self.first = False
                self.compressed = hasattr(msg, "format")
            
            img = (self._compressedImage2CVMat if self.compressed else self._Image2CVMat)(msg)
            return img is not None, img
        else:
            return False, None
        

class ROSCapture(ROSBase):

    def __init__(self, topic):
        super().__init__()
        import rospy
        import rostopic
        from sensor_msgs.msg import CompressedImage, Image
        img_type, *_ = rostopic.get_topic_type(topic)
        self.img = None
        assert img_type.lower().endswith("image")
        self.compressed = img_type.lower().endswith("compressedimage")
        self.updated = False
        rospy.init_node('edgeyolo_detector', anonymous=True)
        rospy.Subscriber(topic, CompressedImage if self.compressed else Image, self.__imageReceivedCallback)

    def __imageReceivedCallback(self, msg):
        self.img = (self._compressedImage2CVMat if self.compressed else self._Image2CVMat)(msg)
        self.updated = True

    def isOpened(self):
        return True
    
    def read(self):
        while not self.updated:
            pass
        self.updated = False
        return True, self.img


def setup_source(args):
    if isinstance(args.source, str) and os.path.isdir(args.source):

        class DirCapture:

            def __init__(self, dir_name):
                self.imgs = []
                for img_type in ["jpg", "png", "jpeg", "bmp", "webp"]:
                    self.imgs += sorted(glob(os.path.join(dir_name, f"*.{img_type}")))

            def isOpened(self):
                return bool(len(self.imgs))

            def read(self):
                print(self.imgs[0])
                now_img = cv2.imread(self.imgs[0])
                self.imgs = self.imgs[1:]
                return now_img is not None, now_img

        source = DirCapture(args.source)
        delay = 0
    else:
        if args.source is not None and args.source.lower().endswith(".bag"):
            cmd_check_topic = f"rosbag info {args.source}"
            if args.topic is None:
                str_show = ""
                topics = []
                count = 0
                for line in os.popen(cmd_check_topic).read().split("topics:")[-1].split("\n"):
                    if len(line):
                        contents = line.split()
                        if contents[-1] not in ["sensor_msgs/CompressedImage", "sensor_msgs/Image"]:
                            continue
                        # print(line)
                        count += 1
                        topics.append(contents[0])
                        str_show += f"{count}. " + topics[-1] + "\n"
                        
                logger.error(f"choose one topic from the following topics:\n{str_show[:-1]}")
                idx = -1
                while True:
                    try:
                        idx = int(input(f"input 1~{count} as your choise and then press enter(-1 to exit):\n>>> "))
                        if idx == -1:
                            break
                        if idx > count or idx < 1:
                            assert False
                        args.topic = topics[idx-1]
                        break
                    except:
                        print("wrong input!")
                        pass
                # return
                if idx == -1:
                    exit(0)
            source = ROSBagCapture(args.source, args.topic)
            delay = 0
        elif args.source is None:
            if args.topic is None:
                str_show = ""
                cmd_check_topic = "rostopic list"
                count = 0
                topics = []
                for line in os.popen(cmd_check_topic).read().split("\n"):
                    if len(line):
                        if os.popen(f"rostopic type {line}").read().split("\n")[0] in ["sensor_msgs/CompressedImage", "sensor_msgs/Image"]:
                            count += 1
                            topics.append(line.split()[0])
                            str_show += f"{count}. " + topics[-1] + "\n"

                logger.error(f"choose one topic from the following topics:\n{str_show[:-1]}")
                while True:
                    try:
                        idx = int(input(f"input 1~{count} as your choise and then press enter(-1 to exit):\n>>> "))
                        if idx == -1:
                            break
                        if idx > count or idx < 1:
                            assert False
                        args.topic = topics[idx-1]
                        break
                    except:
                        print("wrong input!")
                        pass
                # return
                if idx == -1:
                    exit(0)
            source = ROSCapture(args.topic)
            delay = 1
        else:
            source = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
            delay = 1
    return source, delay


def detect_single(args):
    import time
    exist_save_dir = os.path.isdir(args.save_dir)

    # detector setup
    detector = TRTDetector if args.trt else Detector
    detect = detector(
        weight_file=args.weights,
        conf_thres=args.conf_thres,
        nms_thres=args.nms_thres,
        input_size=args.input_size,
        fuse=not args.no_fuse,
        fp16=args.fp16,
        use_decoder=args.use_decoder
    )
    if args.trt:
        args.batch = detect.batch_size

    # source loader setup
    source, delay = setup_source(args)

    all_dt = []
    save_json_path = Path("results")
    save_json_path.mkdir(exist_ok=True)
    all_predictions = []

    dts_len = 300 // args.batch
    success = True

    # ✅ Latency ölçüm değişkenleri
    total_inference_time = 0
    total_total_time = time.time()
    total_frames = 0

    # start inference
    count = 0
    while source.isOpened() and success:
        frames = []
        for _ in range(args.batch):
            success, frame = source.read()
            if not success:
                if not len(frames):
                    cv2.destroyAllWindows()
                    break
                else:
                    while len(frames) < args.batch:
                        frames.append(frames[-1])
            else:
                frames.append(frame)

        if not len(frames):
            break

        # ✅ Inference latency ölçümü
        t_infer_start = time.time()
        if total_frames == 0:
            cv2.imwrite("image.jpeg",frames[0]) 
        results = detect(frames, args.legacy)
        t_infer_end = time.time()
        inference_time = t_infer_end - t_infer_start

        total_inference_time += inference_time
        total_frames += len(frames)

        dt = detect.dt
        all_dt.append(dt)
        if len(all_dt) > dts_len:
            all_dt = all_dt[-dts_len:]
        print(f"\r{dt * 1000 / args.batch:.1f}ms  "
              f"average:{sum(all_dt) / len(all_dt) / args.batch * 1000:.1f}ms", end="      ")

        key = -1

        imgs = draw(deepcopy(frames), results, detect.class_names, 2, draw_label=not args.no_label)

        # Bounding box sonuçlarını JSON için kaydet
        for i, result in enumerate(results):  # her frame için tahmin
            for det in result:  # her bounding box
                x1, y1, x2, y2, conf, cls = map(float, det[:6])
                w = x2 - x1
                h = y2 - y1
                all_predictions.append({
                    "image_id": 0,  # veya filename'den elde edilen ID
                    "bbox": [x1, y1, w, h],
                    "score": conf,
                    "category_id": int(cls)
                })

        for i, img in enumerate(imgs):
            cv2.imshow("EdgeYOLO result", img)
            count += 1

            key = cv2.waitKey(delay)
            if key in [ord("q"), 27]:
                break
            elif key == ord(" "):
                delay = 1 - delay
            elif key == ord("s"):
                if not exist_save_dir:
                    os.makedirs(args.save_dir, exist_ok=True)
                    exist_save_dir = True
                fn = f"{str(date.now()).split('.')[0].replace(':', '').replace('-', '').replace(' ', '')}"
                file_name = fn + ".jpg"
                cv2.imwrite(os.path.join(args.save_dir, file_name), img)
                logger.info(f"image saved to {file_name}.")
        if key in [ord("q"), 27]:
            break

    print()

    # 🖨️ FPS ve latency çıktıları
    total_elapsed = time.time() - total_total_time
    print("\n=== Inference Benchmark Summary ===")
    print(f"Total frames processed: {total_frames}")
    print(f"Average Inference Latency: {1000 * total_inference_time / total_frames:.2f} ms")
    print(f"Average Total Latency: {1000 * total_elapsed / total_frames:.2f} ms")
    print(f"Approx. FPS (including all): {total_frames / total_elapsed:.2f}")

    cv2.imshow("Image", imgs[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Tahmin sonuçlarını diske yaz
    with open(save_json_path / "predictions.json", "w") as f:
        json.dump(all_predictions, f, indent=2)
    logger.info("Tüm tahmin sonuçları 'results/predictions.json' dosyasına kaydedildi.")

    
    

def inference(msg, results, args):
    from edgeyolo.detect import Detector, TRTDetector
    import time

    detector = TRTDetector if args.trt else Detector
    detect = detector(
        weight_file=args.weights,
        conf_thres=args.conf_thres,
        nms_thres=args.nms_thres,
        input_size=args.input_size,
        fuse=not args.no_fuse,
        fp16=args.fp16,
        use_decoder=args.use_decoder
    )
    if args.trt:
        args.batch = detect.batch_size

    source, delay = setup_source(args)

    msg["class_names"] = detect.class_names
    msg["delay"] = delay

    total_inference_time = 0
    total_total_time = time.time()
    total_frames = 0

    while source.isOpened() and not msg["end"]:
        frames = []
        for _ in range(args.batch):
            if msg["end"]:
                break
            success, frame = source.read()
            if not success:
                if not len(frames):
                    break
                while len(frames) < args.batch:
                    frames.append(frames[-1])
            else:
                frames.append(frame)

        if not len(frames):
            break

        if total_frames == 0:
            cv2.imwrite("images.jpeg",frames[0])
        
        t_infer_start = time.time()
        output = detect(frames, args.legacy)
        print("The type of output: ",output)
        t_infer_end = time.time()

        total_inference_time += (t_infer_end - t_infer_start)
        total_frames += len(frames)

        results.put((frames, [r.cpu() for r in output]))

    msg["infer_time"] = total_inference_time
    msg["frame_count"] = total_frames
    msg["total_time"] = time.time() - total_total_time

    torch.cuda.empty_cache()
    msg["end"] = True
    msg["end_count"] += 1



def draw_imgs(msg, results, all_imgs, args):
    from edgeyolo.detect import draw

    while "class_names" not in msg:
        pass
    class_names = msg["class_names"]

    while not msg["end"] or not results.empty():
        # print(len(msg["results"]))
        if not results.empty():
            for img in draw(*results.get(), class_names, 2, draw_label=not args.no_label):
                all_imgs.put(img)
                # print(all_imgs.empty())
    torch.cuda.empty_cache()
    msg["end_count"] += 1


def show(msg, all_imgs, args, pid):
    from time import time
    while "delay" not in msg:
        pass
    delay = msg["delay"]
    exist_save_dir = os.path.isdir(args.save_dir)

        # --- VideoWriter Hazırlığı (MP4 kaydı) ---
    # MP4 için uygun fourcc. 'mp4v' çoğu sistemde çalışır. Olmazsa alternatif: 'avc1' veya 'XVID' + .avi
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = None  # İlk kare geldiğinde başlatacağız

    # Kaydedilecek çıktı dosyası
    os.makedirs(args.save_dir, exist_ok=True)
    video_path = os.path.join(args.save_dir, "output_video.mp4")

    # Yazma FPS'i: args.fps gerçekçi değilse (ör. 99999), 30 kullan.
    write_fps = args.fps if (isinstance(args.fps, int) and args.fps < 1000 and args.fps > 0) else 30

    
    all_dt = []
    t0 = time()
    frame_displayed = 0

    while not msg["end"] or not all_imgs.empty():
        if not all_imgs.empty():
            img = all_imgs.get()
            resized_frame = cv2.resize(img, (1280, 720))
                        # --- İlk karede VideoWriter aç ---
            if out_video is None:
                h, w = resized_frame.shape[:2]
                out_video = cv2.VideoWriter(video_path, fourcc, write_fps, (w, h))

            # --- Her kareyi videoya yaz ---
            out_video.write(resized_frame)

            while time() - t0 < 1. / args.fps - 0.0004:
                pass

            dt = time() - t0
            all_dt.append(dt)
            if len(all_dt) > 300:
                all_dt = all_dt[-300:]

            mean_dt = sum(all_dt) / len(all_dt) * 1000
            print(f"\r{dt * 1000:.1f}ms --> {1. / dt:.1f}FPS, "
                  f"average:{mean_dt:.1f}ms --> {1000. / mean_dt:.1f}FPS", end="      ")

            t0 = time()
            cv2.imshow("EdgeYOLO result", resized_frame)
            frame_displayed += 1

            key = cv2.waitKey(delay)
            if key in [ord("q"), 27]:
                msg["end"] = True
                cv2.destroyAllWindows()
                                # --- Çıkarken writer'ı serbest bırak ---
                if out_video is not None:
                    out_video.release()

                break
            elif key == ord(" "):
                delay = 1 - delay
            elif key == ord("s"):
                if not exist_save_dir:
                    os.makedirs(args.save_dir, exist_ok=True)
                file_name = f"{str(date.now()).split('.')[0].replace(':', '').replace('-', '').replace(' ', '')}.jpg"
                cv2.imwrite(os.path.join(args.save_dir, file_name), img)
                logger.info(f"image saved to {file_name}.")

    print()
    if out_video is not None:
        out_video.release()
    torch.cuda.empty_cache()
    msg["end_count"] += 1

    while not msg["end_count"] == 3:
        pass

    # 🖨️ Latency Raporu (mp mod için)
    print("\n=== Inference Benchmark Summary (Multi-Process Mode) ===")
    total_frames = msg.get("frame_count", 0)
    total_infer_time = msg.get("infer_time", 0)
    total_time = msg.get("total_time", 0)
    if total_frames > 0:
        print(f"Total frames processed: {total_frames}")
        print(f"Average Inference Latency: {1000 * total_infer_time / total_frames:.2f} ms")
        print(f"Average Total Latency: {1000 * total_time / total_frames:.2f} ms")
        print(f"Approx. FPS (including all): {total_frames / total_time:.2f}")

    


    # if platform.system().lower() == "windows":
    #     os.system(F"taskkill /F /PID {pid}")
    # else:
    #     os.system(f"kill -9 {pid}")


def detect_multi(args):
    freeze_support()

    shared_data = Manager().dict()
    shared_data["end"] = False
    shared_data["end_count"] = 0

    results = Manager().Queue()
    all_imgs = Manager().Queue()

    processes = [Process(target=inference, args=(shared_data, results, args)),
                 Process(target=draw_imgs, args=(shared_data, results, all_imgs, args)),
                 Process(target=show, args=(shared_data, all_imgs, args, os.getpid()))]

    [process.start() for process in processes]
    torch.cuda.empty_cache()
    [process.join() for process in processes]


    

if __name__ == '__main__':
    opt = get_args()
    (detect_multi if opt.mp else detect_single)(opt)
    print()


