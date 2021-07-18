import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)

# read data
def get_list(file_name,idx=1):
    dfs = pd.read_csv(file_name, dtype=float)
    data_s = []
    for i, row in dfs.iterrows():
        ts= row[idx]
        data_s.append(ts)
    return data_s

# # fig 3a MR
def draw_3_1a():
    name_list = [ '0.02', '0.1','0.2 ','0.3','0.4']
    num_list1 = [[0.7490153,0.76674175,0.7785593,0.7718064,0.7687113],
                [0.7563309,0.7706809,0.77687114,0.7743388,0.7687113],
                [0.75773776,0.76927406,0.7813731,0.7746202,0.77265054]]
    num_list2 =[[0.7630838,0.7729319,0.77321327,0.77884084,0.7777153],
                [0.7591446,0.7703996,0.7819359,0.77658975,0.77855945],
                [0.7599888,0.771525,0.7816545,0.78418684,0.7774339]]
    num_list3 =[[0.7611143,0.7563309,0.78277993,0.7796849,0.7749015],
                [0.7521103,0.77039963,0.7785594,0.7791221,0.77940357],
                [0.75182885,0.778278,0.7785594,0.77940357,0.776027]]
    num_list1 = np.mean(num_list1, axis=0)
    num_list2 = np.mean(num_list2, axis=0)
    num_list3= np.mean(num_list3, axis=0)
    plt.ylim(0.74, 0.79)
    plt.plot(range(len(num_list1)), num_list1, 'ro-',label='MP-GCN-1')
    plt.plot(range(len(num_list2)), num_list2, 'bo-',label='MP-GCN-1*')
    plt.plot(range(len(num_list3)), num_list3, 'go-', label='MP-GCN-2')
    plt.xticks(range(len(name_list)),name_list)
    plt.legend()
    plt.show()

#fig 3b R8
def draw_3_1b():
    name_list = [ '0.001','0.005','0.01','0.1','0.2']
    num_list1 =[[0.96893543,0.9771584,0.9762447,0.97167647,0.9707627],
    [0.9703059,0.97304696,0.9744174,0.9771584,0.9689355],
    [0.9689355,0.9716764,0.9739606,0.97121966,0.97121954]]
    num_list2 =[[0.9721333,0.96893543,0.9744174,0.97852886,0.9748742],
                [0.97213334,0.97852886,0.9748742,0.9767016,0.9762447],
                [0.97350377,0.9771584,0.9771584,0.9771584,0.97761524]]
    num_list3 =[[0.9716765,0.96756494,0.97304696,0.97259015,0.9707628],\
               [0.9721333,0.97807205,0.9771584,0.97121954,0.97167647],\
               [0.9703059,0.9771584,0.97624475,0.9739606,0.9680218]]
    num_list1 = np.mean(num_list1, axis=0)
    num_list2 = np.mean(num_list2, axis=0)
    num_list3= np.mean(num_list3, axis=0)
    plt.ylim(0.96, 0.985)
    plt.plot(range(len(num_list1)), num_list1, 'ro-',label='MP-GCN-1')
    plt.plot(range(len(num_list2)), num_list2, 'bo-',label='MP-GCN-1*')
    plt.plot(range(len(num_list3)), num_list3, 'go-', label='MP-GCN-2')
    plt.xticks(range(len(name_list)),name_list)
    plt.legend()
    plt.show()

# fig 3a MR multi-head
def draw_3_2a():
    name_list = [ '1', '4','8 ','12','16']
    num_list1 =[[0.7653348,0.77687114,0.77658975,0.7788408,0.7819359],
     [0.76083285,0.77658975,0.776027,0.776027,0.77940345],
     [0.76195836,0.7734947,0.78081036,0.77996624,0.77827805]]
    num_list2 =[[0.77490157,0.7833428,0.7785594,0.78193575,0.7796849],
                [0.77377605,0.7777153,0.77546424,0.7791222,0.7796849],
                [0.77405745,0.78052896,0.78081036,0.7816545,0.7788408]]
    num_list3 =[[0.7647722,0.7779966,0.7830613,0.77321327,0.7765898],
                [0.7633652,0.77771527,0.78137314,0.77405745,0.7712437],
                [0.76505345,0.7802476,0.78503096,0.77574575,0.7751829]]

    num_list1 = np.mean(num_list1, axis=0)
    num_list2 = np.mean(num_list2, axis=0)
    num_list3= np.mean(num_list3, axis=0)
    plt.ylim(0.74, 0.79)
    plt.plot(range(len(num_list1)), num_list1, 'ro-',label='MP-GCN-1')
    plt.plot(range(len(num_list2)), num_list2, 'bo-',label='MP-GCN-1*')
    plt.plot(range(len(num_list3)), num_list3, 'go-', label='MP-GCN-2')
    plt.xticks(range(len(name_list)),name_list)
    plt.legend()
    plt.show()

# fig 3b R8 multi-head
def draw_3_2b():
    name_list = [ '1', '4','8 ','12','16']
    num_list1 =[[0.96299666,0.9757879,0.9767016,0.9739606,0.97852886],
                [0.962083,0.97350377,0.9744174,0.9771584,0.9725901],
                [0.96162623,0.97259015,0.9730469,0.97761524,0.9757879]]
    num_list2=[[0.9680218,0.9744174,0.9771584,0.97350377,0.97898567],
               [0.96893543,0.9716764,0.9744174,0.97852886,0.97944254],
               [0.97121954,0.9767016,0.9751333,0.9789857,0.97761524]]
    num_list3 =[[0.96116936,0.97213334,0.9762447,0.9767016,0.9748742],
                [0.96436715,0.97350377,0.97487426,0.97807205,0.9789857],
                [0.96391034,0.9767016,0.9771584,0.97167647,0.97304696]]
    num_list1 = np.mean(num_list1, axis=0)
    num_list2 = np.mean(num_list2, axis=0)
    num_list3= np.mean(num_list3, axis=0)
    plt.ylim(0.96, 0.985)
    plt.plot(range(len(num_list1)), num_list1, 'ro-',label='MP-GCN-1')
    plt.plot(range(len(num_list2)), num_list2, 'bo-',label='MP-GCN-1*')
    plt.plot(range(len(num_list3)), num_list3, 'go-', label='MP-GCN-2')
    plt.xticks(range(len(name_list)),name_list)
    plt.legend()
    plt.show()

# fig 4a MR
def draw_4a():
    name_list = [ '2.5', '5','10','30','50']
    num_list1 =[0.6222,0.6579,0.6848,0.7409, 0.7545]
    num_list2 = [0.6337, 0.6474, 0.6972, 0.7375, 0.7649]
    num_list3 = [0.6204, 0.6595,  0.6844,  0.7305, 0.7606]
    num_list4 = [0.6284, 0.6445, 0.6864,  0.7200, 0.7404]
    num_list5 = [0.5292, 0.6108, 0.6391, 0.6980, 0.7249]
    num_list6 = [0.5152, 0.5669, 0.6295, 0.6950, 0.7250]
    num_list7 = [0.5717, 0.6067 , 0.6452, 0.6959, 0.7189]
    plt.ylim(0.5, 0.78)

    plt.plot(range(len(num_list1)), num_list1, 'o-',color='darkorange',label='MP-GCN-1')
    plt.plot(range(len(num_list2)), num_list2, 'bo-',label='MP-GCN-1*')
    plt.plot(range(len(num_list3)), num_list3, 'go-', label='MP-GCN-2')
    plt.plot(range(len(num_list4)), num_list4, 'o-',color='red',label='Text-GCN')
    plt.plot(range(len(num_list5)), num_list5, 'co-', label='CNN')
    plt.plot(range(len(num_list6)), num_list6, 'mo-', label='LSTM')
    plt.plot(range(len(num_list7)), num_list7, 'yo-', label='TF-IDF+LR')
    plt.xticks(range(len(name_list)),name_list)
    plt.legend()
    plt.show()

if __name__=='__main__':
    draw_4a()