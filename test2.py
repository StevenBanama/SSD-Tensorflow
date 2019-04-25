#coding=utf-8
import tensorflow as tf
from datasets import dataset_factory
from nets import nets_factory
from preprocessing import ssd_vgg_preprocessing
import cv2
from nets import np_methods
from preprocessing import ssd_vgg_preprocessing

slim = tf.contrib.slim

mapper = {2: "OK", 3:"FIVE", 4: "V", 5:"SIX", 6:"IOU", 7:"GOOD", 8:"BAD"}
SIZE = (160, 160, 3)
MNAME = "mobile_v2_160"
MPATH= "./mobile_v2_224_0225checkpoint"

def init_net():

    with tf.Graph().as_default():
        sess = tf.Session()
        ssd_class = nets_factory.get_network(MNAME)
        ssd_params = ssd_class.default_params._replace(num_classes=21)
        ssd_net = ssd_class(ssd_params)
        img_input = tf.placeholder(tf.uint8, shape=(None, None, 3), name="input_image")

        # Evaluation pre-processing: resize to SSD net shape.
        image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
                img_input, None, None, SIZE[:-1], "NHWC", resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
        image_4d = tf.expand_dims(image_pre, 0)

        ssd_shape = ssd_net.params.img_shape
        ssd_anchors = ssd_net.anchors(ssd_shape)

        arg_scope = ssd_net.arg_scope(data_format="NHWC")
        with slim.arg_scope(arg_scope):
            predictions, localisations, logits, end_points = \
                ssd_net.net(image_4d, is_training=False)
            localisations = ssd_net.bboxes_decode(localisations, ssd_anchors)
            rscore, rbboxes = ssd_net.detected_bboxes(predictions, localisations, select_threshold=0.05, top_k=50, keep_top_k=10, nms_threshold=0.2)


        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(MPATH)

        saver.restore(sess, ckpt.model_checkpoint_path)
        return sess, rscore, rbboxes, bbox_img, img_input, ssd_anchors

def show_cost(func):
    def wrap(*args, **kwargs):
        import time
        from_time = time.time()
        result = func(*args, **kwargs)
        print("cost: ", time.time() - from_time)
        return result 
    return wrap
    
@show_cost
def once_eval(sess, scores, boundboxes, bbox_img, img_input, ssd_anchors, img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    scores, boundboxes, rbbox_img = sess.run([scores, boundboxes, bbox_img], feed_dict={img_input: img})
    keys = scores.keys()
    nums = len(scores[1])
    rclasses, rbboxes = [], []
    for x in xrange(nums):
        print((scores[1].shape, boundboxes[1].shape))
        rc = max([(scores[k][0][x], boundboxes[k][0][x], k) for k in keys], key=lambda cad: cad[0])
        rclasses.append(rc[2]), rbboxes.append(rc[1])
    return rclasses, rbboxes

def test(img_dir, save=False):
    for root, dirlist, files in os.walk(img_dir):
        paths = map(lambda f: os.path.join(root, f), files)
    sess, rscores, boxes, bbox_img, img_input, ssd_anchors = init_net()
    result = {}
    for idx, path in enumerate(paths):
        img = cv2.imread(path)
        (height, width) = img.shape[:-1]
        rclasses, rbboxes = once_eval(sess, rscores, boxes, bbox_img, img_input, ssd_anchors, img)
        for i, box in enumerate(rbboxes):
            ymin, xmin, ymax, xmax = map(int, [box[0] * height, box[1]*width, box[2] * height, box[3] * width])
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
            cv2.putText(img, mapper.get(rclasses[i]), (xmin, ymin), cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255), 1)
        if save:
            cv2.imwrite("./test_out/out%s.jpg"%idx, img)

class GestureDetector:

    def __init__(self, is_train=False):
        if is_train:
            self.tensors = sess, rscores, boxes, bbox_img, img_input, ssd_anchors = init_net()

    def run(self, cv_img):
        rclasses, rbboxes = once_eval(*(self.tensors + (cv_img,)))
        points = []
        height, width = cv_img.shape[:-1]
        for i, box in enumerate(rbboxes):
            if mapper.get(rclasses[i]) <=1:
                continue
            ymin, xmin, ymax, xmax = map(int, [box[0] * height, box[1]*width, box[2] * height, box[3] * width])
            points.append({"x": xmin, "y": ymin, "type": "rect", "width": xmax - xmin, "height": ymax - ymin, "name": mapper.get(rclasses[i])})
        return points

    def gen_model(self, model_path):
        sess, scores, boundboxes, rbbox_img, img_input = self.tensors[:5]
        rclasses, mboxes = [], []
        for k in scores.keys():
            mask = tf.greater(scores[k], 0.3)
            rclasses.append(tf.boolean_mask(tf.cast(mask, tf.int32) * k, mask))
            mboxes.append(tf.boolean_mask(boundboxes[k], mask))

        stack_classes = tf.concat(rclasses, axis=0, name="rclasses")
        stack_boxes = tf.concat(mboxes, axis=0, name="boundboxes")
        from tensorflow.python.framework.graph_util import convert_variables_to_constants
        with sess.graph.as_default():
            print(stack_boxes.get_shape())
            #saver = tf.train.Saver()
            #saver.save(sess, "./gesture_mobile/model.ckpt")
            output_graph_def = convert_variables_to_constants(sess, sess.graph_def, output_node_names=['rclasses', 'boundboxes'])
            with tf.gfile.FastGFile(model_path, mode='wb') as f:
                f.write(output_graph_def.SerializeToString())
        sess.close()

    def reload_pb(self, path='./gesture_mobile/ry_guesture.pb'):
        sess = tf.Session()
        @show_cost
        def run(img):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width = img.shape[:-1]
            img = cv2.resize(img, SIZE[:-1])
            rclasses, rbboxes= sess.run(output, feed_dict={image: img})
            points = []
            for i, box in enumerate(rbboxes):
                ymin, xmin, ymax, xmax = map(int, [box[0] * height, box[1]*width, box[2] * height, box[3] * width])
                points.append({"x": xmin, "y": ymin, "type": "rect", "width": xmax - xmin, "height": ymax - ymin, "name": mapper.get(rclasses[i])})
            return points


        with open(path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            image = tf.placeholder(tf.uint8, SIZE)
            output = tf.import_graph_def(graph_def,
                input_map={'input_image:0': image},
                return_elements=['rclasses:0', "boundboxes:0"])
        return run

def camera_test():
    cap = cv2.VideoCapture(0)
    sess, rscores, boxes, bbox_img, img_input, ssd_anchors = init_net()

    while True:
        ret, cv_image = cap.read()
        print(cv_image.shape)
        img = cv2.resize(cv_image, (640, 480))
        (height, width) = img.shape[:-1]

        if not ret:
            return -1
        rclasses, rbboxes = once_eval(sess, rscores, boxes, bbox_img, img_input, ssd_anchors, img)
        print(img.shape)
        print(rclasses, rbboxes)
        for i, box in enumerate(rbboxes):
            ymin, xmin, ymax, xmax = map(int, [box[0] * height, box[1]*width, box[2] * height, box[3] * width])
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
            cv2.putText(img, mapper.get(rclasses[i]), (xmin, ymin), cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255), 1)
        cv2.imshow("result", img)

        if cv2.waitKey(3) == 27:
            break
    cap.release()

import os
if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    camera_test()
    #test("./validation")
    #GestureDetector().gen_model("gesture_v0224.pb") 
