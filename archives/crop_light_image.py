def crop_light_image(self):
    rec0 = self.raw_corners[0].lu
    rec2 = self.raw_corners[2].lu
    w, h = (self.raw_corners[0].ru[0] - self.raw_corners[0].lu[0],
            self.raw_corners[0].ld[1] - self.raw_corners[0].lu[1])

    x0, y0 = map(int, (rec0[0] + 1 / 2 * w, rec0[1] - 3 / 2 * h))  # light 左上角
    x1, y1 = map(int, (rec2[0] + 1 / 2 * w, rec2[1] - 1 / 8 * h))  # light 右下角

    # self._debug(self.raw_im[y0:y1, x0:x1])

    y0, y1 = 0 if y0 < 0 else y0, 0 if y1 < 0 else y1

    return self.raw_im[y0:y1, x0:x1]
