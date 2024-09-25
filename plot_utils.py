import os
from PIL import Image
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np


def plot_training_loss(perf_loss, epochs, networkInfo, base_path=None):
    """
    Hàm vẽ và lưu đồ thị mất mát huấn luyện.

    :param perf_loss: Danh sách các giá trị mất mát huấn luyện theo từng epoch.
    :param epochs: Số lượng epochs.
    :param networkInfo: Thông tin về mạng để sử dụng trong tiêu đề biểu đồ.
    :param base_path: Đường dẫn cơ bản cho thư mục lưu trữ hình ảnh.
    """

    if base_path is None:
        base_path = os.getcwd()
    # create dirs
    plot_root_path = base_path + '/viz/'
    plot_path = plot_root_path + networkInfo
    if not os.path.exists(plot_root_path):
        os.makedirs(plot_root_path)
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    # plot
    epoch = list(range(epochs))
    training_avg = []
    cnt = 0.0
    counter = 0
    for i in perf_loss:
        counter += 1
        cnt += i
        training_avg.append(cnt / counter)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss for ' + networkInfo)
    plt.scatter(epoch, perf_loss, c='b', marker='o', s=25)
    epoch_np = np.array(epoch)
    avg_np = np.array(training_avg)
    # Dùng spline nếu có đủ điểm dữ liệu (tối thiểu 3 điểm để spline hoạt động)
    if len(epoch_np) > 3 and len(set(avg_np)) > 3:
        epoch_new = np.linspace(epoch_np.min(), epoch_np.max(), 300)
        avg_smooth = make_interp_spline(epoch_np, avg_np)(epoch_new)
        plt.plot(epoch_new, avg_smooth, c='r')
    else:
        # Nếu không đủ điểm cho spline, dùng đường thẳng nối giữa các điểm
        plt.plot(epoch_np, avg_np, c='r', label='Avg. Training Loss')
    my_x_ticks = np.arange(0, epochs, 1)
    plt.xticks(my_x_ticks)
    # save plot
    plt.savefig(plot_path + '/' + 'training_loss.png', bbox_inches='tight', dpi=300)

def plot_accuracy(perf_train_acc, perf_val_acc, epochs, networkInfo, base_path=None):
    if base_path is None:
        base_path = os.getcwd()
    # create dirs
    plot_root_path = base_path + '/viz/'
    plot_path = plot_root_path + networkInfo
    if not os.path.exists(plot_root_path):
        os.makedirs(plot_root_path)
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    # plot the accuracies v.s. epoch number
    epoch = list(range(epochs))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train / Val Accuracy for ' + networkInfo)
    plt.plot(epoch, perf_train_acc, marker='o', markersize=5, label='train')
    plt.plot(epoch, perf_val_acc, c='orange', marker='o', markersize=5, label='val')
    # plt.plot(epoch, test_acc, c='red', marker='o', markersize=5, label='test')
    my_x_ticks = np.arange(0, epochs, 1)
    plt.xticks(my_x_ticks)
    plt.legend(loc='lower right')
    # save plot
    plt.savefig(plot_path + '/' + 'accuracy.png', bbox_inches='tight', dpi=300)