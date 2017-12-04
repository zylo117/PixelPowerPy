from read_bin.preprocess import preprocess
from read_bin.conv2d_matlab import conv2
import numpy as np
from scipy.ndimage.filters import correlate
import numba


def dp(IDraw, bayerformat="rggb", pedestal=64, bitdepth=10, threshold_defect=0.19, threshold_defectLow=0.12,
       threshold_detectable=32,
       cluster_type="bayer", cluster_size=3, neighbour_type="avg", more_precise=False):
    if threshold_defect > 1:
        ID = preprocess(IDraw, bayerformat, outputformat="raw", mode=0, bitdepth=10, pedestal=64, FOV=0,
                        whitebalance=False, signed=True, more_precise=more_precise)
    else:
        ID = preprocess(IDraw, bayerformat, outputformat="raw", mode=0, bitdepth=10, pedestal=64, FOV=0,
                        whitebalance=True, signed=True, more_precise=more_precise)

    h = ID.shape[0]
    w = ID.shape[1]

    # 初始化变量
    count_cluster = 0
    count_DP = 0
    count_NDP = 0
    count_NDPP = 0
    roi_size = 15

    # 对四个边缘进行填补（延展长度roiSize / 2)
    ID_mirror = np.pad(ID, roi_size - 1, "symmetric")

    h_mirror = ID_mirror.shape[0]
    w_mirror = ID_mirror.shape[1]

    # 对奇偶行进行交换
    ID_mirror[[1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12], :] = ID_mirror[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], :]
    ID_mirror[[-13, -14, -11, -12, -9, -10, -7, -8, -5, -6, -3, -4, -1, -2], :] = ID_mirror[[-14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1], :]
    # 对奇偶列进行交换
    ID_mirror[:, [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12]] = ID_mirror[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]
    ID_mirror[:, [-13, -14, -11, -12, -9, -10, -7, -8, -5, -6, -3, -4, -1, -2]] = ID_mirror[:, [-14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1]]

    # 把图像进行均值归一化
    ID_avg = np.zeros(ID_mirror.shape)
    kernel = np.zeros((2 * roi_size - 1, 2 * roi_size - 1))
    kernel[::2, ::2] = 1 / (roi_size ** 2)
    ID_avg = correlate(ID_mirror, kernel)
    # 去除多余边框，恢复原分辨率
    ID_avg = ID_avg[roi_size - 1: h_mirror - roi_size + 1, roi_size - 1: w_mirror - roi_size + 1]

    # 找出defective pixels
    if threshold_defect > 1:
        ID_delta = ID - ID_avg  # LSB（Least significant bit） 值，最小有效值，黑场（dark field）测试条件
        map_defect = (np.abs(ID_delta) > threshold_defect).astype(np.double)  # 求出大于threshold的布尔坐标图，转换为double（0或1）
    else:
        # 计算差异百分比
        ID_percDiff = np.abs((ID - ID_avg) / ID_avg)
        ID_percDiff[np.isnan(ID_percDiff)] = 0  # 把所有0/0的无效值替换为0
        ID_delta = ID_percDiff
        map_defect = (np.abs(ID_delta) > threshold_defect).astype(np.double)  # 求出大于threshold的布尔坐标图，转换为double（0或1）
        ID_delta = ID_delta * 100  # 小数转百分比

    # 定义kernel

    # 1.簇（cluster）内核
    # 当内核对应点的值，大于100 + cluster_size - 1的时候，该点就被标记为cluster的中心。后期增长（grow）cluster的时候，内核会被再次应用
    if cluster_type is "bayer":
        cluster_pattern = np.array([[1, 0, 1, 0, 1],
                                   [0, 0, 0, 0, 0],
                                   [1, 0, 100, 0, 1],
                                   [0, 0, 0, 0, 0],
                                   [1, 0, 1, 0, 1]])
        
    elif cluster_type is "raw":
        cluster_pattern = np.array([[1, 1, 1, 1, 1],
                                   [1, 1, 1, 1, 1],
                                   [1, 1, 100, 1, 1],
                                   [1, 1, 1, 1, 1],
                                   [1, 1, 1, 1, 1]])

    # 2.梯对（ladder pair）内核
    ladder_pattern = np.array([[0, 1, 0, 1, 0],
                              [0, 0, 0, 0, 0],
                              [0, 1, 33, 1, 0],
                              [0, 0, 0, 0, 0],
                              [0, 1, 0, 1, 0]])

    # 3.对（pair）内核
    pair_pattern = np.array([[1, 0, 1, 0, 1],
                            [0, 0, 0, 0, 0],
                            [1, 0, 33, 0, 1],
                            [0, 0, 0, 0, 0],
                            [1, 0, 1, 0, 1]])

    # 4.行（row）内核
    row_pattern = np.array([[0, 0, 0, 0, 0],
                           [1, 1, 1, 1, 1],
                           [0, 0, 33, 0, 0],
                           [1, 1, 1, 1, 1],
                           [0, 0, 0, 0, 0]])

    # 1.簇（cluster）检测
    # 检测cluster defects，并标记cluster里面的所有像素
    map_temp_cluster = conv2(map_defect, cluster_pattern)  # 应用卷积
    map_temp_cluster = (np.abs(map_temp_cluster) >= (100 + cluster_size - 1)).astype(np.double)
    map_temp_cluster = conv2(map_temp_cluster, cluster_pattern)  # 再次应用卷积，找出影响区
    map_temp_cluster = map_temp_cluster * map_defect  # 标出影响区里面的所有像素（看看defective pixels是否与cluster有交集）
    map_temp_cluster = (map_temp_cluster > 0).astype(np.double)

    # 筛选出无cluster的区域（排除cluster的影响进行下一步检测）
    map_no_cluster = map_defect - map_temp_cluster

    # 2.边界2行2列的defects检测
    map_temp_border = np.ones(ID.shape)
    map_temp_border[2:h - 2, 2:w - 2] = 0
    map_temp_border = map_no_cluster * map_temp_border

    """
    怕不是DP Tagging修补dp的源码，但是缺乏defectivePixel_featureScore_Apple，暂时无法使用
     remove all NVM locations featureScoreスキップのため無効化
     the code below will iterate through all NVM locations and set those as non-defects
    for i=1:size(nvm,1)
        nvm_x =  nvm(i,1);
        nvm_y =  nvm(i,2);
		if (nvm_y >= 3) && (nvm_y <= (h-2)) && (nvm_x >=3) && (nvm_x <= (w-2))
			kernel =  mapTemp_cluster((nvm_y-2):(nvm_y+2),(nvm_x-2):(nvm_x+2));
		else
			kernel = zeros(5,5);
		end
		 Check for valid image coordinates
		if ((nvm_y >=1) && (nvm_y <= h) && (nvm_x >= 1) && (nvm_x <=w))
			 Check for tagged cluster
	       if (defectivePixel_featureScore_Apple(kernel, 2) <= nvm_clusterSize)
    	        mapTemp_cluster(nvm_y, nvm_x) = 0;
        	end
	        map_noCluster(nvm_y,nvm_x) = 0;
    	else 
    		 Do nothing
    	end    
    end
    """

    # 把非cluster的DP标记出来，从而标记DP/DPP/NDP/NDPP
    # Detectable Pixel/Detectable Pixel Pair/Detectable Pixel Pair/Non-Detectable Pixel Pair
    (temp_y, temp_x) = np.where(map_no_cluster > 0)

    map_temp_detection = np.zeros(ID.shape)
    for i in range(0, len(temp_y)):
        # 提取周围的3x3同色区域，注意，同色
        bayer_neighbour3_y = np.arange(temp_y[i] - 2, temp_y[i] + 3, 2)
        bayer_neighbour3_x = np.arange(temp_x[i] - 2, temp_x[i] + 3, 2)
        # 创建坐标对
        bayer_neighbour = np.transpose(np.vstack((np.tile(bayer_neighbour3_y, 3), np.repeat(bayer_neighbour3_x, 3))))
        bayer_neighbour = np.delete(bayer_neighbour, 4, axis=0)
        bayer_neighbour = bayer_neighbour[bayer_neighbour[:, 0] >= 0, :]
        bayer_neighbour = bayer_neighbour[bayer_neighbour[:, 0] < h, :]
        bayer_neighbour = bayer_neighbour[bayer_neighbour[:, 1] >= 0, :]
        bayer_neighbour = bayer_neighbour[bayer_neighbour[:, 1] < w, :]
        # 获取对应像素值
        temp_ROI = np.sort(ID[bayer_neighbour[:, 0], bayer_neighbour[:, 1]])

        if neighbour_type is "avg":
            temp_ROI = temp_ROI[1:-1] # 去除极大极小值
            temp_avg = np.mean(temp_ROI)
            temp_neighbour = np.abs(ID[temp_y[i], temp_x[i]] - temp_avg)
        elif neighbour_type is "delta":
            temp_diff = np.abs(temp_ROI - ID[temp_y, temp_x])
            temp_neighbour = np.min(temp_diff)

        # 根据差异来判断
        if temp_neighbour > threshold_detectable:
            map_temp_detection[temp_y[i], temp_x[i]] = 3
        else:
            map_temp_detection[temp_y[i], temp_x[i]] = 1

    # 标记DP/DPP/NDP/NDPP/Border defects
    map_temp_conv = conv2(map_temp_detection, pair_pattern)
    map_temp_DP = (map_temp_conv == 99).astype(np.double)  # 浮点布尔图
    map_temp_NDP = (map_temp_conv == 33).astype(np.double)  # 浮点布尔图
    map_temp_DPP = ((map_temp_conv != 99) * (map_temp_conv > 35)).astype(np.double)  # 浮点布尔图，DPP中心
    map_temp_NDPP = (map_temp_conv == 34).astype(np.double)  # 浮点布尔图，NDPP中心

    # 标记DLP/NLP(NDLP)
    # Detectable Ladder Pixel/ Non-detectable Ladder Pixel
    map_temp_conv = conv2(map_temp_detection, ladder_pattern)
    map_temp_DLP = ((map_temp_conv != 99) * (map_temp_conv > 35)).astype(np.double)  # 浮点布尔图
    map_temp_NLP = (map_temp_conv == 34).astype(np.double)  # 浮点布尔图

    # 标记ARPD(Adjacent row pair defects)
    if cluster_type is "bayer" and threshold_defect >= 1:
        map_temp_conv = np.zeros(ID.shape)
    else:
        map_temp_conv = conv2(map_temp_detection, row_pattern)

    map_temp_ARPD = ((map_temp_conv > 33) * (map_temp_conv != 99)).astype(np.double)  # 浮点布尔图，标记ARPD的中心

    # 标记low contrast cluster低对比度簇（仅限光场）
    map_temp_clusterlow = np.zeros(ID.shape)
    if threshold_defect <= 1:  # 光场条件下
        map_defect_low = (np.abs(ID_delta) > (threshold_defectLow * 100)).astype(np.double)
        map_defect_low = map_defect_low - map_defect

        # 当内核对应点的值，大于100 + cluster_size - 1的时候，该点就被标记为clusterlow的中心。后期增长（grow）cluster的时候，内核会被再次应用
        cluster_pattern = np.array([[1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1],
                                    [1, 1, 100, 1, 1],
                                    [1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1]])

        # 找出低对比度簇
        map_temp_clusterlow = conv2(map_defect_low, cluster_pattern)
        map_temp_clusterlow = (map_temp_clusterlow >= (100 + cluster_size - 1)).astype(np.double)
        map_temp_clusterlow = conv2(map_temp_clusterlow, cluster_pattern)
        map_temp_clusterlow = map_temp_clusterlow * map_defect_low
        map_temp_clusterlow = (map_temp_clusterlow > 0).astype(np.double)

    """
    Feature功能尚未加入
    """

    mapFail = np.max(np.array((map_temp_DP,
                               2 * map_temp_DPP,
                               3 * map_temp_border,
                               # 4 * map_temp_feature,
                               5 * map_temp_NDP,
                               6 * map_temp_NDPP,
                               7 * map_temp_DLP,
                               8 * map_temp_NLP,
                               9 * map_temp_ARPD,
                               10 * map_temp_cluster,
                               11 * map_temp_clusterlow)), axis=0)

    dp_fail_result = []

    DP_data = [np.where(mapFail == 1), len((np.where(mapFail == 1))[0])]
    DPP_data = [np.where(mapFail == 2), len((np.where(mapFail == 2))[0]) / 2]
    NDP_data = [np.where(mapFail == 5), len((np.where(mapFail == 5))[0])]
    NDPP_data = [np.where(mapFail == 6), len((np.where(mapFail == 6))[0]) / 2]
    DLP_data = [np.where(mapFail == 7), len((np.where(mapFail == 7))[0]) / 2]
    NLP_data = [np.where(mapFail == 8), len((np.where(mapFail == 8))[0]) / 2]
    feature_data = [np.where(mapFail == 4), len((np.where(mapFail == 4))[0])]
    ARPD_data = [np.where(mapFail == 9), len((np.where(mapFail == 9))[0]) / 2]
    cluster_data = [np.where(mapFail == 10), len((np.where(mapFail == 10))[0])]
    clusterlow_data = [np.where(mapFail == 11), len((np.where(mapFail == 11))[0])]
    border_data = [np.where(mapFail == 3), len((np.where(mapFail == 3))[0])]
    all_dp_data = [np.where(mapFail > 0), len((np.where(mapFail > 0))[0])]

    dp_fail_result = [DP_data, DPP_data, NDP_data, NDPP_data, DLP_data, NLP_data,feature_data, ARPD_data, cluster_data, clusterlow_data, border_data, all_dp_data]

    dp_pointset = []
    for i in range(len(dp_fail_result[-1][0][0])):
        dp_pointset.append((dp_fail_result[-1][0][1][i], dp_fail_result[-1][0][0][i]))

    return dp_fail_result, dp_pointset, ID


import cv2
import imutils


def draw_defective_pixel(dp_fail_result, draw_on=None):
    if draw_on is None:
        background = (np.ones((2340, 3856)) * 255).astype(np.uint8)
        background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
    else:
        background = (draw_on / 4).astype(np.uint8)
        background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)

    # cv2.circle(background, (300, 300), 10, (222, 0, 222))

    pointset = dp_fail_result[-1][0]
    for i in range(len(pointset[0])):
        cv2.circle(background, (pointset[1][i], pointset[0][i]), 100, (222, 0, 222), 10)

    background = imutils.resize(background, width=512)
    cv2.imshow("DP", background)
    cv2.waitKey()
