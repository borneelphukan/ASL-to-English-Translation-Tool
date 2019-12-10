import cv2
import numpy as np
from keras.models import load_model


# loading the weights & specifying image size
model = load_model('Weights.h5')
image_x, image_y = 64, 64


def prediction():
    import numpy as np                                          # debugged import
    from keras.preprocessing import image
    test_image = image.load_img('test.png', target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)

    if result[0][0] == 1:
        return 'A'
    elif result[0][1] == 1:
        return 'B'
    elif result[0][2] == 1:
        return 'C'
    elif result[0][3] == 1:
        return 'D'
    elif result[0][4] == 1:
        return 'E'
    elif result[0][5] == 1:
        return 'F'
    elif result[0][6] == 1:
        return 'G'
    elif result[0][7] == 1:
        return 'H'
    elif result[0][8] == 1:
        return 'I'
    elif result[0][9] == 1:
        return 'J'
    elif result[0][10] == 1:
        return 'K'
    elif result[0][11] == 1:
        return 'L'
    elif result[0][12] == 1:
        return 'M'
    elif result[0][13] == 1:
        return 'N'
    elif result[0][14] == 1:
        return 'O'
    elif result[0][15] == 1:
        return 'P'
    elif result[0][16] == 1:
        return 'Q'
    elif result[0][17] == 1:
        return 'R'
    elif result[0][18] == 1:
        return 'S'
    elif result[0][19] == 1:
        return 'T'
    elif result[0][20] == 1:
        return 'U'
    elif result[0][21] == 1:
        return 'V'
    elif result[0][22] == 1:
        return 'W'
    elif result[0][23] == 1:
        return 'X'
    elif result[0][24] == 1:
        return 'Y'
    elif result[0][25] == 1:
        return 'Z'


video = cv2.VideoCapture(0)


def nothing(x):
    pass


cv2.namedWindow("trackbars")
cv2.resizeWindow("trackbars", 400, 300)
cv2.createTrackbar("HUE (L)", "trackbars", 0, 255, nothing)
cv2.createTrackbar("SATURATION (L)", "trackbars", 0, 255, nothing)
cv2.createTrackbar("VALUE (L)", "trackbars", 0, 255, nothing)
cv2.createTrackbar("HUE (U)", "trackbars", 255, 255, nothing)
cv2.createTrackbar("SATURATION (U)", "trackbars", 255, 255, nothing)
cv2.createTrackbar("VALUE (U)", "trackbars", 255, 255, nothing)

#cv2.namedWindow("Output")
image_text = ''

while True:
    ret, frame = video.read()
    frame = cv2.flip(frame, 1)
    hue_l = cv2.getTrackbarPos("HUE (L)", "trackbars")
    sat_l = cv2.getTrackbarPos("SATURATION (L)", "trackbars")
    val_l = cv2.getTrackbarPos("VALUE (L)", "trackbars")
    hue_u = cv2.getTrackbarPos("HUE (U)", "trackbars")
    sat_u = cv2.getTrackbarPos("SATURATION (U)", "trackbars")
    val_u = cv2.getTrackbarPos("VALUE (U)", "trackbars")

    image = cv2.rectangle(frame, (425, 100), (625, 300), (0, 255, 255), thickness=2, lineType=8, shift=0)

    low_bound = np.array([hue_l, sat_l, val_l])
    high_bound = np.array([hue_u, sat_u, val_u])
    image_crop = image[102:298, 427:623]
    hsv = cv2.cvtColor(image_crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, low_bound, high_bound)

    cv2.putText(frame, image_text, (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 3.5, (0, 255, 0))
    cv2.imshow("Test Output", frame)
    cv2.imshow("HSV Output", mask)

    image_name = "test.png"
    save_image = cv2.resize(mask, (image_x, image_y))
    cv2.imwrite(image_name, save_image)
    image_text = prediction()

    if cv2.waitKey(1) == 27:
        break

video.release()
cv2.destroyAllWindows()
