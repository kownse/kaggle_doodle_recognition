import os
import datetime as dt
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import numpy as np
import json
import keras
import math
import glob
import random
from sklearn.cluster import KMeans

BASE_SIZE = 256
NCATS = 340

def plotDoodles(x):
    plt.figure()
    cnt = x.shape[0]
    side = int(math.sqrt(cnt))
    fig, axs = plt.subplots(side, side, figsize=(10,15))
    for i in range(side):
        for j in range(side):
            ax = axs[i,j] 
            ax.imshow(x[i * side + j].squeeze())
            ax.axis('off')
    plt.show()

def crop_center(image_data):
    non_empty_columns = np.where(image_data.max(axis=0)>0)[0]
    non_empty_rows = np.where(image_data.max(axis=1)>0)[0]
    cropBox = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))

    image_data_new = image_data[cropBox[0]:cropBox[1]+1, cropBox[2]:cropBox[3]+1]
    return image_data_new

def add_padding(img, pad_l, pad_t, pad_r, pad_b):
    height, width = img.shape
    #Adding padding to the left side.
    pad_left = np.zeros((height, pad_l), dtype = np.int)
    img = np.concatenate((pad_left, img), axis = 1)

    #Adding padding to the top.
    pad_up = np.zeros((pad_t, pad_l + width))
    img = np.concatenate((pad_up, img), axis = 0)

    #Adding padding to the right.
    pad_right = np.zeros((height + pad_t, pad_r))
    img = np.concatenate((img, pad_right), axis = 1)

    #Adding padding to the bottom
    pad_bottom = np.zeros((pad_b, pad_l + width + pad_r))
    img = np.concatenate((img, pad_bottom), axis = 0)

    return img

def center_image(imgnew, size):
    imgcent = imgnew
    pad_up = 0
    pad_down = 0
    pad_left = 0
    pad_right = 0
    
    if imgnew.shape[0] < size:
        pad_up = math.ceil((size - imgnew.shape[0]) / 2)
        pad_down = math.floor((size - imgnew.shape[0]) / 2)
        
    if imgnew.shape[1] < size:
        pad_left = math.ceil((size - imgnew.shape[1]) / 2)
        pad_right = math.floor((size - imgnew.shape[1]) / 2)
    
    imgcent = add_padding(imgnew, pad_left, pad_up, pad_right, pad_down)

    return imgcent

def pad_center(image_data, size):
    imgnew = crop_center(image_data)
#     print(imgnew.shape)
    imgnew = center_image(imgnew, size)
    return imgnew

def draw_cv2(raw_strokes, size=256, lw=6, time_color=True, center = True):
    img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)
    for t, stroke in enumerate(raw_strokes):
        color_step = int(200 / len(raw_strokes))
        for i in range(len(stroke[0]) - 1):
#             color = 255 - min(t, 10) * 13 if time_color else 255
            color = 255 - color_step * t if time_color else 255
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                         (stroke[0][i + 1], stroke[1][i + 1]), color, lw)
            
    if size != BASE_SIZE:
        img = cv2.resize(img, (size, size))
        
    if center:
        img = pad_center(img, size)
        
    return img

def draw_cv2_pointcnts(raw_strokes, size=256, lw=2, center = True):
    points_cnt = [len(s[0]) for s in raw_strokes]
    min_cnt = min(points_cnt)
    max_cnt = max(points_cnt)
    if max_cnt > min_cnt:
        color_step = (200 / (max_cnt - min_cnt))
    else:
        color_step = 0
#     print(max_cnt, min_cnt, color_step)
    
    img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)
    
    for t, stroke in enumerate(raw_strokes):
        if color_step > 0:
            color = 255 - int(color_step * (max_cnt - len(stroke[0])))
            #print(color, color_step, len(stroke[0]), (len(stroke[0]) - min_cnt))
        else:
            color = 255
        
        for i in range(len(stroke[0]) - 1):
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                         (stroke[0][i + 1], stroke[1][i + 1]), color, lw)
    if size != BASE_SIZE:
        img = cv2.resize(img, (size, size))
    
    if center:
        img = pad_center(img, size)
        
    return img
    
def get_line_length(stroke):
    length = 0
    for i in range(len(stroke[0]) - 1):
        x = stroke[0][i + 1] - stroke[0][i]
        y = stroke[1][i + 1] - stroke[1][i]
        step_length = (x**2 + y**2) **(1/2)
#         print(step_length)
        length += step_length
    return int(length)

def draw_cv2_linelength(raw_strokes, size=256, lw=2, center = True):
    lengths = [get_line_length(s) for s in raw_strokes]
    min_length = min(lengths)
    max_length = max(lengths)
    if max_length > min_length:
        color_step = 200 / (max_length - min_length)
    else:
        color_step = 0
#     print(max_cnt, min_cnt, color_step)
    
    img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)
    
    for t, stroke in enumerate(raw_strokes):
        if color_step > 0:
            color = 255 - int(color_step * (max_length - get_line_length(stroke)))
#             print(color, color_step, get_line_length(stroke))
        else:
            color = 255
        
        for i in range(len(stroke[0]) - 1):
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                         (stroke[0][i + 1], stroke[1][i + 1]), color, lw)
    if size != BASE_SIZE:
        img = cv2.resize(img, (size, size))
    
    if center:
        img = pad_center(img, size)
        
    return img
    
def draw_cv2_whole(raw_strokes, size=256, lw=2, center = True):
    img = np.zeros((size, size, 3), np.uint8)
    img[..., 0] = draw_cv2(raw_strokes, size=size, lw=lw, time_color=True, center = center)
    img[..., 1] = draw_cv2_pointcnts(raw_strokes, size=size, lw=lw, center = center)
    img[..., 2] = draw_cv2_linelength(raw_strokes, size=size, lw=lw, center = center)
    
    return img

def draw_cv2_parts(raw_strokes, size=256, lw=2, center = False, detail_threshold = 0):
    img = np.zeros((size, size, 3), np.uint8)
    img[..., 0] = draw_cv2(raw_strokes, size=size, lw=lw, time_color=True, center = center)
    
    lengths = [get_line_length(s) for s in raw_strokes]
    mean_length = np.mean(np.array(lengths))
    point_cnts = [len(s[0]) for s in raw_strokes]
    mean_pcnts = np.mean(np.array(point_cnts))
    
    outlines = []
    details = []
    for s in raw_strokes:
        if get_line_length(s) < mean_length or len(s[0]) < mean_pcnts:
            details.append(s)
        else:
            outlines.append(s)
        
    img[..., 1] = draw_cv2(outlines, size=size, time_color = False, center = center)
    if len(details) > 0:
        img[..., 2] = draw_cv2(details, size=size, time_color = False, center = center)
    
    return img

def draw_cv2_parts_opt(raw_strokes, size=256, lw=2, center = True):
    img = np.zeros((BASE_SIZE, BASE_SIZE, 3), np.uint8)
    
    lengths = np.array([get_line_length(s) for s in raw_strokes])
    mean_length = int(np.mean(lengths))
    
    point_cnts = np.array([len(s[0]) for s in raw_strokes])
    mean_pcnts = int(np.mean(point_cnts))
    
    cnt_stroke = len(raw_strokes)

    not_one_stroke = cnt_stroke > 1 and np.any(lengths != lengths[0])
#     print(not_one_stroke)
    if not_one_stroke:
        outlines = []
        details = []
        for idx,s in enumerate(raw_strokes):
            line_length = get_line_length(s)
            line_ptr = len(s[0])
            if line_length < mean_length:# or line_ptr < mean_pcnts:
                details.append(line_length)
            else:
                outlines.append(line_length)

        details = np.array(details)
        outlines = np.array(outlines)
        
#         print('outline', outlines)
#         print('details', details)

        max_outline = np.max(outlines)
        min_outline = np.min(outlines)
        len_outline = (max_outline - min_outline) if len(outlines) > 1 else max_outline
        len_outline = max_outline if len_outline == 0 else len_outline

        if len(details) > 0:
            max_detail = np.max(details)
            min_detail = np.min(details)
            len_detail = (max_detail - min_detail) if len(details) > 1 else max_detail
            len_detail = 1 if len_detail == 0 else len_detail
        else:
            max_detail = 0
            min_detail = 0
            len_detail = 1
#         print(len(raw_strokes), len_outline, len_detail)

    color_step = int(200 / cnt_stroke)
    for idx,s in enumerate(raw_strokes):
        color_r = 255 - color_step * idx
        
        if not_one_stroke:
            line_length = get_line_length(s)
            line_ptr = len(s[0])
            if line_length < mean_length:# or line_ptr < mean_pcnts:
                color_g = 0
                color_b = int(200 * (1 - (max_detail - line_length) / len_detail)) + 55
            else:
                color_g = int(200 * (1 - (max_outline - line_length) / len_outline)) + 55
                color_b = 0
        else:
            color_g = 255
            color_b = 0
        
        color = (color_r, color_g, color_b)
#         print(color)
        for i in range(len(s[0]) - 1):
            _ = cv2.line(img, (s[0][i], s[1][i]),
                             (s[0][i + 1], s[1][i + 1]), color, lw)
    if size != BASE_SIZE:
        img = cv2.resize(img, (size, size))
        
    return img

def draw_cv2_parts_threshold(raw_strokes, size=256, lw=2, center = False, detail_threshold = 5):
    img = np.zeros((size, size, 3), np.uint8)
    img[..., 0] = draw_cv2(raw_strokes, size=size, lw=lw, time_color=True, center = center)
    
    if len(raw_strokes) >= detail_threshold:
        lengths = [get_line_length(s) for s in raw_strokes]
        mean_length = np.mean(np.array(lengths))
        point_cnts = [len(s[0]) for s in raw_strokes]
        mean_pcnts = np.mean(np.array(point_cnts))

        outlines = []
        details = []
        for s in raw_strokes:
            if get_line_length(s) < mean_length or len(s[0]) < mean_pcnts:
                details.append(s)
            else:
                outlines.append(s)

        img[..., 1] = draw_cv2(outlines, size=size, time_color = False, center = center)
        if len(details) > 0:
            img[..., 2] = draw_cv2(details, size=size, time_color = False, center = center)
    else:
        img[..., 1] = draw_cv2(raw_strokes, size=size, time_color = False, center = center)
    
    return img

def mixup_onedata(data, labels, weight, index, batch_size):
    x = np.zeros_like(data, dtype=data.dtype)
    y = np.zeros_like(labels, dtype=labels.dtype)
    
    x1, x2 = data, data[index]
    y1, y2 = labels, labels[index]
    
    for i in range(batch_size):
        x[i] = x1[i] * weight[i] + x2[i] * (1 - weight[i])
        y[i] = y1[i] * weight[i] + y2[i] * (1 - weight[i])
    return x, y

def mixup_all(data, labels, alpha):
    batch_size = len(labels)
    weight = np.random.beta(alpha, alpha, batch_size)
    index = np.random.permutation(batch_size)
    
    return mixup_onedata(data, labels, weight, index, batch_size)
    
def image_generator_xd(size, batchsize, lw=2, 
                       df_path = '../input/train_all.csv', time_color=True, preprocess_input = None,
                       channel = 1, mixup = 0, center = False):
    while True:
        for df in pd.read_csv(df_path, chunksize=batchsize):
            df['drawing'] = df['drawing'].apply(json.loads)
            x = np.zeros((len(df), size, size, channel), dtype=np.uint8)
            for i, raw_strokes in enumerate(df.drawing.values):
                if channel == 1:
                    x[i, :, :, 0] = draw_cv2(raw_strokes, size=size, 
                                             lw=lw,
                                             channel = channel, center = center)
                else:
                    x[i, :, :, :] = draw_cv2_parts(raw_strokes, size=size, 
                                             lw=lw, center = center)
            
            if 'word' in df:
                y = keras.utils.to_categorical(df.word, num_classes=NCATS)

                if mixup > 0:
                    x, y = mixup_all(x, y, mixup)

                if preprocess_input is not None:
                    x = preprocess_input(x.astype(np.float32)).astype(np.float32)

                yield x, y  
            else:
                if preprocess_input is not None:
                    x = preprocess_input(x.astype(np.float32)).astype(np.float32)
                yield x
            
def df_to_image_array_xd(df, size, lw=2, 
                         time_color=True, preprocess_input = None,
                         channel = 1, center = False):
    df['drawing'] = df['drawing'].apply(json.loads)
    x = np.zeros((len(df), size, size, channel ), dtype=np.uint8)
    for i, raw_strokes in enumerate(df.drawing.values):
        if channel == 1:
            img = draw_cv2(raw_strokes, size=size, 
                                     lw=lw, center = center)
#             print(img.shape)
            x[i, :, :, 0] = img
        else:
            x[i, :, :, :] = draw_cv2_parts(raw_strokes, size=size, 
                                     lw=lw, center = center)
    if preprocess_input is not None:
        print('x shape',x.shape, 'x max', x.max())
        x = preprocess_input(x.astype(np.float32)).astype(np.float32)
    return x 

def image_generator_xd_parts_threshold(size, batchsize, lw=2, 
                       df_path = '../input/train_all.csv', time_color=True, preprocess_input = None,
                       channel = 1, mixup = 0, center = False, detail_threshold = 5):
    while True:
        for df in pd.read_csv(df_path, chunksize=batchsize):
            df['drawing'] = df['drawing'].apply(json.loads)
            x = np.zeros((len(df), size, size, channel), dtype=np.uint8)
            for i, raw_strokes in enumerate(df.drawing.values):
                if channel == 1:
                    x[i, :, :, 0] = draw_cv2(raw_strokes, size=size, 
                                             lw=lw,
                                             channel = channel, center = center)
                else:
                    x[i, :, :, :] = draw_cv2_parts_threshold(raw_strokes, size=size, 
                                             lw=lw, center = center,
                                             detail_threshold = detail_threshold)
            
            if 'word' in df:
                y = keras.utils.to_categorical(df.word, num_classes=NCATS)

                if mixup > 0:
                    x, y = mixup_all(x, y, mixup)

                if preprocess_input is not None:
                    x = preprocess_input(x.astype(np.float32)).astype(np.float32)

                yield x, y  
            else:
                if preprocess_input is not None:
                    x = preprocess_input(x.astype(np.float32)).astype(np.float32)
                yield x
            
def df_to_image_array_xd_parts_threshold(df, size, lw=2, 
                         time_color=True, preprocess_input = None,
                         channel = 1, center = False, detail_threshold = 5):
    df['drawing'] = df['drawing'].apply(json.loads)
    x = np.zeros((len(df), size, size, channel ), dtype=np.uint8)
    for i, raw_strokes in enumerate(df.drawing.values):
        if channel == 1:
            img = draw_cv2(raw_strokes, size=size, 
                                     lw=lw, center = center)
#             print(img.shape)
            x[i, :, :, 0] = img
        else:
            x[i, :, :, :] = draw_cv2_parts_threshold(raw_strokes, size=size, 
                                     lw=lw, center = center,
                                     detail_threshold = detail_threshold)
    if preprocess_input is not None:
        print('x shape',x.shape, 'x max', x.max())
        x = preprocess_input(x.astype(np.float32)).astype(np.float32)
    return x 

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def get_id_by_path(path):
    return cat2id[path_leaf(path)[:-4]]

def get_color_from_lst(lst, maxlight = True):
    min_value = min(lst)
    max_value = max(lst)
    colors = []
    for value in lst:
        if max_value == min_value:
            color = 255
        else:
            if maxlight == False:
                color = int((max_value - value) / (max_value - min_value) * 200 + 55)
            else:
                color = int((value - min_value) / (max_value - min_value) * 200 + 55)
        colors.append(color)
    return colors

def get_scale_simple(stroke, pad = 5):
    maxx = max([max(s[0]) for s in stroke])
    maxy = max([max(s[1]) for s in stroke])
    minx = min([min(s[0]) for s in stroke])
    miny = min([min(s[1]) for s in stroke])

    width = maxx - minx
    height = maxy - miny
#     print(width, height)
    scale = BASE_SIZE / (width + 2 * pad) if width > height else BASE_SIZE / (height + 2 * pad)
    return scale, minx, miny

def draw_time_encoding(stroke, size=256, lw = 2):
    BASE_SIZE = 256
    pad = 5
    timestamp = []
    draw_time = []
    pause_time = []
    for idx,s in enumerate(stroke):
        timestamp.append(s[2][0])
        draw_time.append(s[2][-1] - s[2][0])
        if idx > 0:
            pause_time.append(stroke[idx][2][0] - stroke[idx-1][2][-1])
    if len(pause_time) > 0:
        pause_time = [min(pause_time)] + pause_time     
    else:
        pause_time = [0]

    drawtime_color = get_color_from_lst(draw_time)
    pausetime_color = get_color_from_lst(pause_time)
    timestamp_color = get_color_from_lst(timestamp, maxlight=False)
    
    scale, minx, miny = get_scale_simple(stroke, pad)
    
    img = np.zeros((BASE_SIZE, BASE_SIZE, 3), np.uint8)
    stroke_num = len(stroke)
    for idx,s in enumerate(stroke):
        time_start_color = timestamp_color[idx]
        if idx == stroke_num - 1:
            time_end_color = 55
        else:
            time_end_color = timestamp_color[idx + 1]
            
        ptr_num = len(s[0])
        for i in range(ptr_num - 1):
            x0 = int((s[0][i] - minx + pad) * scale)
            y0 = int((s[1][i] - miny + pad) * scale)
            x1 = int((s[0][i+1] - minx + pad) * scale)
            y1 = int((s[1][i + 1] - miny + pad) * scale)
            
            ratio = i / ptr_num
            timecolor = int(time_start_color * (1 - ratio) + time_end_color * ratio)
            color = (timecolor, drawtime_color[idx], pausetime_color[idx])
            cv2.line(img, (x0, y0), (x1, y1), color, lw)
    
    if size != BASE_SIZE:
        img = cv2.resize(img, (size, size))
    return img

def image_detail_generator(size, batchsize, lw = 6, preprocess_input = None):
    print('gen inited')
    np_classes = np.load('../input/classes.npy')
    cat2id = {cat.replace(' ', '_'):k for k, cat in enumerate(np_classes)}

    files = glob.glob('../input/train_raw/*.csv')

    readers = []
    for f in files:
        readers.append(pd.read_csv(f, chunksize=1))

    while True:
        x = np.zeros((batchsize, size, size, 3), dtype=np.uint8)
        y = np.zeros(batchsize, dtype=np.uint32)
        for i in range(batchsize):
            idx = random.randint(0, len(readers) - 1)
#             if i == 0:
#                 print('ramdom idx', idx)
            line = next(readers[idx])
#             imgpath = '/media/HDD/kaggle/doodle/time_img/{}.png'.format(line.key_id.values[0])
#             if os.path.exists(imgpath):
#                 img = cv2.imread(imgpath)
#             else:
#                 stroke = json.loads(line['drawing'].values[0])
#                 img = draw_time_encoding(stroke, size=size, lw=lw)
#                 cv2.imwrite(imgpath, img)
            
            stroke = json.loads(line['drawing'].values[0])
            img = draw_time_encoding(stroke, size=size, lw=lw)

            id = cat2id[line['word'].values[0].replace(' ', '_')]
            x[i,...] = img
            y[i] = id
            
        if preprocess_input is not None:
            x = preprocess_input(x.astype(np.float32)).astype(np.float32)
        y = keras.utils.to_categorical(y, num_classes=NCATS)
        
        yield x, y
        
def df_to_image_array_timeencoding(df, size, lw=6, preprocess_input = None):
    df['drawing'] = df['drawing'].apply(json.loads)
    x = np.zeros((len(df), size, size, 3 ), dtype=np.uint8)
    for i, raw_strokes in enumerate(df.drawing.values):
        img = draw_time_encoding(raw_strokes, size=size, lw=lw)
        x[i, ...] = img
    if preprocess_input is not None:
        print('x shape',x.shape, 'x max', x.max())
        x = preprocess_input(x.astype(np.float32)).astype(np.float32)
    return x 

def draw_drawtime_parts(stroke, size=256, lw = 2):
#     print(lw)
    BASE_SIZE = 256
    pad = 5
    timestamp = []
    draw_time = []
    for idx,s in enumerate(stroke):
        timestamp.append(s[2][0])
        draw_time.append(abs(s[2][-1] - s[2][0]))
    
    if len(stroke) > 1:
        log_draw_time = [math.log1p(t) for t in draw_time]
        X = np.array(log_draw_time).reshape(-1,1)
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(X)
        y_kmeans = kmeans.predict(X)
        draw_type = list(y_kmeans)

        drawtime0 = []
        drawtime1 = []
        for idx,tag in enumerate(draw_type):
            if tag == 0:
                drawtime0.append(draw_time[idx])
            else:
                drawtime1.append(draw_time[idx])
        mean0 = np.mean(drawtime0)
        mean1 = np.mean(drawtime1)

        if mean0 > mean1:
            outline = 0
            detail = 1
        else:
            outline = 1
            detail = 0
    else:
        draw_type = [0]
        outline = 0
        detail = 1

    timestamp_color = get_color_from_lst(timestamp, maxlight=False)
    scale, minx, miny = get_scale_simple(stroke, pad)

    img = np.zeros((BASE_SIZE, BASE_SIZE, 3), np.uint8)
    stroke_num = len(stroke)
    for idx,s in enumerate(stroke):
        time_start_color = timestamp_color[idx]
        if idx == stroke_num - 1:
            time_end_color = 55
        else:
            time_end_color = timestamp_color[idx + 1]

        ptr_num = len(s[0])
        for i in range(ptr_num - 1):
            x0 = int((s[0][i] - minx + pad) * scale)
            y0 = int((s[1][i] - miny + pad) * scale)
            x1 = int((s[0][i+1] - minx + pad) * scale)
            y1 = int((s[1][i + 1] - miny + pad) * scale)

            ratio = i / ptr_num
            timecolor = int(time_start_color * (1 - ratio) + time_end_color * ratio)
            color = (timecolor, 
                     255 if draw_type[idx] == outline else 0, 
                     255 if draw_type[idx] == detail else 0)
            cv2.line(img, (x0, y0), (x1, y1), color, lw)
            
    if size != BASE_SIZE:
        img = cv2.resize(img, (size, size))
        
    return img

def image_detail_generator_temp(size, batchsize, drawfunc, lw = 6, preprocess_input = None):
    print('gen inited')
    np_classes = np.load('../input/classes.npy')
    cat2id = {cat.replace(' ', '_'):k for k, cat in enumerate(np_classes)}

    files = glob.glob('../input/train_raw/*.csv')

    readers = []
    for f in files:
        readers.append(pd.read_csv(f, chunksize=1))

    while True:
        x = np.zeros((batchsize, size, size, 3), dtype=np.uint8)
        y = np.zeros(batchsize, dtype=np.uint32)
        for i in range(batchsize):
            idx = random.randint(0, len(readers) - 1)
#             if i == 0:
#                 print('ramdom idx', idx)
            line = next(readers[idx])
            imgpath = '/media/HDD/kaggle/doodle/drawtime_img/{}.png'.format(line.key_id.values[0])
            if os.path.exists(imgpath):
#                 print(imgpath, 'exist')
                img = cv2.imread(imgpath)
            else:
#                 print(imgpath, 'not exist')
                stroke = json.loads(line['drawing'].values[0])
                img = drawfunc(stroke, size=size, lw=lw)
                cv2.imwrite(imgpath, img)

            id = cat2id[line['word'].values[0].replace(' ', '_')]
            x[i,...] = img
            y[i] = id
            
        if preprocess_input is not None:
            x = preprocess_input(x.astype(np.float32)).astype(np.float32)
        y = keras.utils.to_categorical(y, num_classes=NCATS)
        
        yield x, y
        
def df_to_image_array_temp(df, size, drawfunc, lw=6, preprocess_input = None):
    df['drawing'] = df['drawing'].apply(json.loads)
    x = np.zeros((len(df), size, size, 3 ), dtype=np.uint8)
    for i, raw_strokes in enumerate(df.drawing.values):
        img = drawfunc(raw_strokes, size=size, lw=lw)
        x[i, ...] = img
    if preprocess_input is not None:
        print('x shape',x.shape, 'x max', x.max())
        x = preprocess_input(x.astype(np.float32)).astype(np.float32)
    return x 