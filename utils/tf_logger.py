# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch.utils.tensorboard import SummaryWriter
import numpy as np


class Logger(object):
    
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""
        for i, img in enumerate(images):
            self.writer.add_image(f'{tag}/{i}', img, step, dataformats='HWC')

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""
        self.writer.add_histogram(tag, values, step, bins=bins)
        self.writer.flush()
