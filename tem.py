import os
import numpy as np
import cv2
import xml.etree.ElementTree as ET

def hough_line(img, degree, degree2):
    # Rho and Theta ranges
    thetas = np.deg2rad(np.arange(degree, degree2))
    height, width = img.shape
    diag_len = int(np.ceil(np.sqrt(width * width + height * height)))  # max_dist
    rhos = np.linspace(-diag_len, diag_len, int(diag_len * 2) + 1)  # create rhos length of a diagonal

    # store them to not calculate everytime
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    accumulator = np.zeros((int(diag_len * 2), num_thetas), dtype=np.uint64)
    y_idxs, x_idxs = np.nonzero(img)  # (row, col) indexes to edges

    # Voting is started here for each rho and theta pairs
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        for t_idx in range(num_thetas):
            # Calculate rho. diag_len is added for a positive index
            rho = int(round(x * cos_t[t_idx] + y * sin_t[t_idx]) + int(diag_len))
            accumulator[rho, t_idx] += 1
    return accumulator, thetas, rhos


def line_intersect(Ax1, Ay1, Ax2, Ay2, Bx1, By1, Bx2, By2):
    d = (By2 - By1) * (Ax2 - Ax1) - (Bx2 - Bx1) * (Ay2 - Ay1)
    if d:
        uA = ((Bx2 - Bx1) * (Ay1 - By1) - (By2 - By1) * (Ax1 - Bx1)) / d
        uB = ((Ax2 - Ax1) * (Ay1 - By1) - (Ay2 - Ay1) * (Ax1 - Bx1)) / d
    else:
        return
    if not (0 <= uA <= 1 and 0 <= uB <= 1):
        return
    x = Ax1 + uA * (Ax2 - Ax1)
    y = Ay1 + uA * (Ay2 - Ay1)

    return x, y


def intersect(img, v1, h1):
    x1, y1 = line_intersect(v1[0], v1[1], v1[2], v1[3], h1[0], h1[1], h1[2], h1[3])
    cv2.circle(img, (round(x1), round(y1)), radius=0, color=(0, 0, 170), thickness=10)
    return round(x1), round(y1)


def get_lines(l, until, bounds, sortby, r, t):
    vert_1 = list()  # remove so close lines each other
    if (until > 0):
        l = l[:until]
    else:
        l = l[until:]
    for i in l:
        y = i[0]
        x = i[1]
        rho = r[y]
        theta = t[x]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = (a * rho)
        y0 = (b * rho)
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        # the code below is written for the redundant duplicate lines
        if len(vert_1) == 0:
            vert_1.append([x1, y1, x2, y2])
        for del_rr in range(len(vert_1)):
            if (-bounds < vert_1[del_rr][sortby] - x1 < bounds) and (
                    -bounds < vert_1[del_rr][sortby + 2] - x2 < bounds):
                break
            elif (del_rr == (len(vert_1) - 1)):
                vert_1.append([x1, y1, x2, y2])
                cv2.line(img0, (x1, y1), (x2, y2), (255, 255, 255), 2)
    vert_1.sort(reverse=True, key=lambda x: x[sortby] + x[sortby + 2])
    return vert_1


def bb_intersection_over_union(boxA, boxB):

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])#get the coordinates
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)#these are the 2 boxes areas
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)#ground truth and predicted
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


path = "C:\\Users\\AtakanAYYILDIZ\\PycharmProjects\\untitled8\\images"
Canny_edge_path = "C:\\Users\\AtakanAYYILDIZ\\PycharmProjects\\untitled8\\Canny"
output_path = "C:\\Users\\AtakanAYYILDIZ\\PycharmProjects\\untitled8\\Output"
only_plate = "C:\\Users\\AtakanAYYILDIZ\\PycharmProjects\\untitled8\\Plates"
annotation_path = "C:\\Users\\AtakanAYYILDIZ\\PycharmProjects\\untitled8\\annotations"
anno_counter = 0
iou_list=list()
for file in os.listdir(path):
    file_path = path + "\\" + file
    img0 = cv2.imread(file_path)
    img = cv2.imread(file_path, 0)
    imgGry = cv2.medianBlur(img, 3)
    edges = cv2.Canny(imgGry, 150, 200, apertureSize=3)  # 240,350
    cv2.imwrite(os.path.join(Canny_edge_path,file),edges)

    ####first verticals
    l = list()
    acc, t, r = hough_line(edges, -5, 0)
    for y in range(acc.shape[0]):
        for x in range(acc.shape[1]):  # displays only the first line
            if 20 < acc[y][x] < 30:
                x = list([y, x, acc[y][x]])
                l.append(x)
    l.sort(key=lambda x: x[2])
    verticals_1 = get_lines(l=l, until=10, bounds=10, sortby=0, r=r, t=t)  # sortby=0 sort by x

    ####second verticals
    l = list()
    acc, t, r = hough_line(edges, 180, 185)
    for y in range(acc.shape[0]):
        for x in range(acc.shape[1]):  # displays only the first line
            if 18 < acc[y][x] < 30:
                x = list([y, x, acc[y][x]])
                l.append(x)
    l.sort(key=lambda x: x[2])  # sort by votes
    verticals_2 = get_lines(l=l, until=10, bounds=10, sortby=0, r=r, t=t)  # sortby=0 sort by x

    ###horizontal lines
    l = list()
    acc, t, r = hough_line(edges, 0, 180)
    for y in range(acc.shape[0]):
        for x in range(acc.shape[1]):  # displays only the first line
            if 50 < acc[y][x] and t[x] < 1.6:
                l.append([y, x, acc[y][x]])
    l.sort(key=lambda x: x[2])
    horizontals = get_lines(l=l, until=-15, bounds=15, sortby=1, r=r, t=t)  # sortby=1 sort by y

    h1 = horizontals[0]
    for h in horizontals:  # find horizontals
        if h1[1] - h[1] > 20 and h1[3] - h[3] > 20:  # plates are usually not close to each other
            h2 = h
            break
    v2 = min(verticals_1[-3], verticals_2[-3])  # find verticals
    v1 = max(verticals_2[1], verticals_1[1])

    px1, py1 = intersect(img0, v1, h1)
    px2, py2 = intersect(img0, v1, h2)  # find intersection to create rectangle for the plate
    px3, py3 = intersect(img0, v2, h1)
    px4, py4 = intersect(img0, v2, h2)

    minx = min(px1, px2, px3, px4)
    maxx = max(px1, px2, px3, px4)  # calculate the iuo
    miny = min(py1, py2, py3, py4)
    maxy = max(py1, py2, py3, py4)

    anno_file = "\\Cars" + str(anno_counter) + ".xml"
    anno_tree = ET.parse(annotation_path + anno_file)
    anno_root = anno_tree.getroot()
    annos = list()
    for ann in anno_root[4]:
        for i in ann:
            annos.append(int(i.text))
    xml_minx = annos[0]
    xml_miny = annos[1]
    xml_maxx = annos[2]
    xml_maxy = annos[3]
    iou = bb_intersection_over_union([minx,miny,maxx,maxy], annos)
    print("{}: {:.4f}".format(file, iou))
    iou_list.append(iou)
    cv2.line(img0, (px1, py1), (px2, py2), (228, 0, 0), 5)
    cv2.line(img0, (px1, py1), (px3, py3), (228, 0, 0), 5)
    cv2.line(img0, (px4, py4), (px2, py2), (228, 0, 0), 5)#create rectangle
    cv2.line(img0, (px3, py3), (px4, py4), (228, 0, 0), 5)
    cv2.imwrite(os.path.join(output_path, file), img0)
    cv2.imwrite(os.path.join(only_plate,file),img0)
    cv2.waitKey(0)
average_iou=sum(iou_list)*100/len(iou_list)
print("Average of iuo is: %{:.4f}".format(average_iou))