import cv2
import tensorflow as tf

def show_cost(func):
    def wrap(*args, **kwargs):
        import time
        from_time = time.time()
        result = func(*args, **kwargs)
        print("cost: ", time.time() - from_time)
        return result 
    return wrap

mapper = {2: "OK", 3:"FIVE", 4: "V", 5:"SIX", 6:"IOU", 7:"GOOD", 8:"BAD"}
SIZE = (160, 160, 3)
MNAME = "mobile_v2_160"
MPATH= "./mobile_v2_224_0225checkpoint"


class GestureDetector:
    def __init__(self):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


    @show_cost
    def run(self, img):
        with self.sess.as_default():
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width = img.shape[:-1]
            img = cv2.resize(img, SIZE[:-1])
            print("coming")
            rclasses, rbboxes = self.sess.run(self.output, feed_dict={self.image: img})
            print("end")
            points = []
            for i, box in enumerate(rbboxes):
                ymin, xmin, ymax, xmax = map(int, [box[0] * height, box[1]*width, box[2] * height, box[3] * width])
                points.append({"x": xmin, "y": ymin, "type": "rect", "width": xmax - xmin, "height": ymax - ymin, "name": mapper.get(rclasses[i])})
            return points


    def reload_pb(self, path='./gesture_mobile/ry_guesture.pb'):
        with self.sess.as_default():
            with open(path, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                self.image = tf.placeholder(tf.uint8, SIZE)
                self.output = tf.import_graph_def(graph_def,
                input_map={'input_image:0': self.image},
                return_elements=['rclasses:0', "boundboxes:0"])

