import cv2
import numpy as np
from skimage.morphology import area_opening


class PictureProcess:

    def dilatation(self, img):
        kernel = np.ones((2, 2), np.uint8)
        dilation = cv2.dilate(img, kernel, 1)
        return dilation

    def process(self, img):
        if img.std() < 30:
            wariki = cv2.inRange(img, (0, 0, 0), (20, 20, 20)) + cv2.inRange(img, (210, 210, 210), (255, 255, 255))
            wariki = self.dilatation(self.dilatation(area_opening(self.dilatation(wariki))))
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, wariki = cv2.threshold(gray, 27, 255, cv2.THRESH_BINARY_INV)
            wariki = self.dilatation(self.dilatation(area_opening(self.dilatation(wariki))))

            totalLabels, label_ids, values, centroid = cv2.connectedComponentsWithStats(wariki, 4, cv2.CV_32S)

            areas = []
            for i in range(1, totalLabels):
                areas.append(values[i, cv2.CC_STAT_AREA])
            output = np.zeros(wariki.shape, dtype="uint8")
            new_areas = np.array(areas)
            new_areas = new_areas[(new_areas > 1) & (new_areas < 600)]
            median = np.median(new_areas)
            for i, area in enumerate(areas):
                if median - 180 < area < median + 180:
                    componentMask = (label_ids == (i + 1)).astype("uint8") * 255
                    output = cv2.bitwise_or(output, componentMask)

            wariki = area_opening(output[60:wariki.shape[0] - 60, 60:wariki.shape[1] - 60], 100)

        totalLabels, label_ids, values, centroid = cv2.connectedComponentsWithStats(wariki, 4, cv2.CV_32S)

        output = np.zeros(wariki.shape, dtype="uint8")
        for i in range(1, totalLabels):
            x, y = int(centroid[i][0]), int(centroid[i][1])
            output = cv2.circle(output, (x, y), 5, (255, 255, 255), 10)

        (totalLabels, label_ids, values, centroid) = cv2.connectedComponentsWithStats(wariki,
                                                                                      4,
                                                                                      cv2.CV_32S)

        if totalLabels - 1 > 10:
            cls = 'II' if totalLabels - 1 <= 12 else 'III'
            return output, cls

        counter = 0
        for i in range(1, totalLabels):
            (X, Y) = centroid[i]

            y_l, y_r = max(int(Y) - 300, 0), min(int(Y) + 300, wariki.shape[0])
            x_l, x_r = max(int(X) - 300, 0), min(int(X) + 300, wariki.shape[1])

            (count, _, _, _) = cv2.connectedComponentsWithStats(wariki[y_l:y_r, x_l:x_r], 4, cv2.CV_32S)
            counter += count - 1

        cls = 'I' if counter >= 60 else 'VI'
        return output, cls