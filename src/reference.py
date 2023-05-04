import numpy as np


def addPadding(xyz_rangeImg, padding):
    height, width, dim = xyz_rangeImg.shape[0], \
                         xyz_rangeImg.shape[1], \
                         xyz_rangeImg.shape[2]
    if padding > 0:
        xyz_rangeImg_pad = np.zeros(shape=(height + padding * 2, width + padding * 2, dim))
        for i in range(0, height):
            for j in range(0, width):
                xyz_rangeImg_pad[i + padding, j + padding, :] = xyz_rangeImg[i, j, :]

    return xyz_rangeImg_pad


def mask_gen(xyz_rangeImg):
    mask = xyz_rangeImg[:, :, 0:1]
    mask[mask == np.nan] = False
    mask[mask == 0] = False
    mask[mask != False] = True
    return mask


def myCluster(xyz_rangeImg, mask, th_H=0.5, th_V=1.0, window_size=np.array([7, 7])):

    height, width, dim = xyz_rangeImg.shape[0], xyz_rangeImg.shape[1], xyz_rangeImg.shape[2]

    # mask = mask_gen(xyz_rangeImg=xyz_rangeImg)
    mask = np.array(mask)
    mask = mask.reshape(64, 2048, 1)
    xyzvl_rangeImg = np.concatenate((xyz_rangeImg, mask, np.zeros(shape=(height, width, 1))), axis=2)
    dim = dim + 2

    maxlabel_lines = np.ones(shape=64)
    valid_ring = np.zeros(shape=64)
    label_img = np.zeros(shape=(height, width))

    # % ----------------------- first run channel-wise labeling -----------------------

    point_neighbor = xyz_rangeImg[0,0,:]
    point_first = xyz_rangeImg[0,0,:]
    point_last = xyz_rangeImg[0,0,:]

    for i in range(0, height):
        valid_first = False
        # maxlabel = 1
        for j in range(0, width):
            # get the current point
            point_current = xyzvl_rangeImg[i, j, :]

            if point_current[3] == 0:
                continue

            # init the neighbor point, note the first point
            # if not valid_neighbor:
            #     valid_neighbor = True
            # point_neighbor = xyzvl_rangeImg[i, j, :]
            # maxlabel = 1
            # xyzvl_rangeImg[i, j, 4] = maxlabel
            if not valid_first:
                point_first = point_current
                point_neighbor = point_current
                valid_first = True

            # group the line wise point
            # if valid_neighbor:
            d = np.linalg.norm(point_current[0:3] - point_neighbor[0:3])
            if d <= th_H:
                xyzvl_rangeImg[i, j, 4] = maxlabel_lines[i]
            else:
                maxlabel_lines[i] = maxlabel_lines[i] + 1
                xyzvl_rangeImg[i, j, 4] = maxlabel_lines[i]

            # % update the point neighbor and point last
            point_neighbor = xyzvl_rangeImg[i, j, :]
            point_last = xyzvl_rangeImg[i, j, :]

        # % note the maxlabel for each line and ring connection
        d = np.linalg.norm(point_first[0:3] - point_last[0:3])
        if d <= th_H:
            valid_ring[i] = True
        else:
            valid_ring[i] = False

    # Accumulate the labels
    start_label = 0
    for i in range(0, height):
        if valid_ring[i]:
            maxlabel_lines[i] = maxlabel_lines[i] - 1

        for j in range(0, width):
            if (xyzvl_rangeImg[i, j, 3] == 0):
                continue

            if valid_ring[i]:
                if xyzvl_rangeImg[i, j, 4] > maxlabel_lines[i]:
                    xyzvl_rangeImg[i, j, 4] = 1

            xyzvl_rangeImg[i, j, 4] = xyzvl_rangeImg[i, j, 4] + start_label

        start_label = start_label + maxlabel_lines[i]
    
    # Check the middle label
    # with open('test.txt', 'w') as outfile:
    #     # I'm writing a header here just for the sake of readability
    #     # Any line starting with "#" will be ignored by numpy.loadtxt
    #     outfile.write('# Array shape: {0}\n'.format(xyzvl_rangeImg[:,:,4].shape))
        
    #     # Iterating through a ndimensional array produces slices along
    #     # the last axis. This is equivalent to data[i,:,:] in this case
    #     for data_slice in xyzvl_rangeImg[:,:,4]:

    #         # The formatting string indicates that I'm writing out
    #         # the values in left-justified columns 7 characters in width
    #         # with 2 decimal places.  
    #         np.savetxt(outfile, data_slice.T, fmt='%-7.2f')

    #         # Writing out a break to indicate different slices...
    #         outfile.write('# New slice\n')


    # % ----------------------- second run label merging -----------------------
    maxlabel = np.max(xyzvl_rangeImg[:, :, 4])
    # print(maxlabel)
    window_width = (window_size - 1) / 2
    window_width = window_width.astype(int)
    stride = np.array([1, 1])
    pair_last = np.array([1, 1])

    # % padding the input frame
    padding = int(window_width[0])
    xyz_rangeImg_pad = addPadding(xyzvl_rangeImg, padding)

    # init the merge variables
    label_current = 1
    labelsToMerge = []
    mergeTable = np.arange(0, maxlabel + 1)

    # % sliding window
    for h in range(0, height):
        for w in range(0, width):
            if mask[h,w] == 0 and w != width-1:
                continue
            # % update the window
            # % flag_outRange = false;
            window = np.zeros(shape=(window_size[0], window_size[1], dim))
            for i in range(0, window_size[0]):
                for j in range(0, window_size[1]):
                    # offsetY = h + i - 1
                    # offsetX = w + j - 1
                    offsetY = h + i
                    offsetX = w + j                    
                    # if(offsetX>0 && offsetY>0 && offsetY<height && offsetX<width)
                    window[i, j, :] = xyz_rangeImg_pad[offsetY, offsetX, :]
            # %                     else
            # %                         window(i,j,:) = zeros(1,1,dim);
            # %                         window(i,j,dim) = -1;
            # %                         flag_outRange = true;
            # %                     end

            #        % find the merge label pair
            #        % get the up and down lines
            # point_current = window[window_width[0], window_width[1], :]
            point_current = xyzvl_rangeImg[h,w,:]
            # if point_current[3] == 0:
            #     continue
            # end
            # % update the labels if current label change or end of frame
            # if (label_current != point_current[4] or (h == height-1 and w == width-1)):
            if label_current != point_current[4] or w == width-1:
                if len(labelsToMerge) != 0:
                    # % Sort and unique
                    labelsToMerge = np.unique(labelsToMerge)
                    # % update the merge Table
                    for n in range(0, len(labelsToMerge)):
                        mergeTable[labelsToMerge[n]] = mergeTable[labelsToMerge[0]]

                # % update the label_current
                # print(label_current, point_current[4])
                label_current = point_current[4]
                labelsToMerge = []


            # lines = np.concatenate((window[0:window_width[0], :, :],
            #                         window[window_width[0] + 1:window_size[0], :, :]),
            #                         axis=0)

            lines = window[0:window_width[0], :, :]

            # for i in range(0, window_size[0] - 1):
            for i in range(0, window_width[0]):
                for j in range(0, window_size[1]):
                    point_neighbor = lines[i, j, :]
                    if point_neighbor[3] == 0:
                        continue

                    d = np.linalg.norm(point_current[0:3] - point_neighbor[0:3])
                    if d < th_V:
                        # pair = np.sort([point_current[4], point_neighbor[4]]).astype(int)
                        pair = np.array([ mergeTable[point_current[4].astype(int)] , mergeTable[point_neighbor[4].astype(int)] ]).astype(int) 

                        # if not np.array_equal(pair, pair_last):
                        #     labelsToMerge.append(pair[0])
                        #     labelsToMerge.append(pair[1])

                        #     pair_last = pair

                        if np.size (np.where(labelsToMerge == pair[0])) == 0:
                            labelsToMerge.append(pair[0])
                        if np.size (np.where(labelsToMerge == pair[1])) == 0:
                            labelsToMerge.append(pair[1])
                    else:
                        continue

    
                        
    for h in range(0, height):
        for w in range(0, width):
            if xyzvl_rangeImg[h, w, 3]:
                outLabel = mergeTable[int(xyzvl_rangeImg[h, w, 4])] .astype(int)
                outLabel = mergeTable[outLabel].astype(int)
                label_img[h, w]  = outLabel

                # label_img[h, w] = mergeTable[int(xyzvl_rangeImg[h, w, 4])].astype(int)
    return label_img.astype(int)