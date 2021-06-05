import tensorflow as tf
import argparse

if __name__ == '__main__':
    # First, export to onnx , and then run onnx-tf convert -i depth_completion.onnx -o depth_completion.pb to convert to TF models!!!

    parser = argparse.ArgumentParser()
    parser.add_argument('--load-model', type=str,
                        default='/Users/user/PycharmProjects/DepthCompletion/tmp/depth_completion.pb')
    parser.add_argument('--export-path', type=str,
                        default='/Users/user/PycharmProjects/DepthCompletion/tmp/depth_completion.tflite')
    args = parser.parse_args()

    # make a converter object from the saved tensorflow file
    converter = tf.lite.TFLiteConverter.from_saved_model(args.load_model)
    # tell converter which type of optimization techniques to use
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # to view the best option for optimization read documentation of tflite about optimization
    # go to this link https://www.tensorflow.org/lite/guide/get_started#4_optimize_your_model_optional

    # convert the model
    tf_lite_model = converter.convert()
    # save the converted model
    open(args.export_path, 'wb').write(tf_lite_model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float32]
    tf_lite_model = converter.convert()
    open(args.export_path + 'fp32', 'wb').write(tf_lite_model)
