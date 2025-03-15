import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard


class CustomTensorBoard(TensorBoard):
    '''
    Provide manual logging for custom metrics
    '''
    def __init__(self, log_dir, **kwargs):
        super().__init__(log_dir=log_dir, **kwargs)
        self.writer = tf.summary.create_file_writer(log_dir)

    def set_model(self, model):
        pass  # Prevents auto-logging of model graphs

    def log(self, step, **stats):
        with self.writer.as_default():
            for name, value in stats.items():
                tf.summary.scalar(name, value, step=step)
            self.writer.flush()  # Ensure real-time logging, prevent data loss if program ends unexpectedly