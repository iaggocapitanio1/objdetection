import cv2
import numpy


def detect_corners(path: str, threshold=None, f_xy=None, switch_var: bool = False, **kwargs):
    if threshold is None:
        threshold = [100, 100]
    if f_xy is None:
        f_xy = [0.5, 0.5]
    kernel = numpy.ones((5, 5))
    obj_width = None
    obj_height = None
    img = cv2.imread(path)

    while True:
        key = cv2.waitKey(0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (5, 5), 1)
        img = cv2.Canny(img, threshold1=threshold[0], threshold2=threshold[1])
        # img = numpy.float32(img)
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = cv2.dilate(img, kernel, iterations=3)
        img = cv2.erode(img, kernel, iterations=2)
        contours, hierarchy, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # dst = cv2.cornerHarris(img, 2, 3, 0.04)
        # dst = cv2.dilate(dst, numpy.ones((3, 3)), iterations=5)
        # img[dst > 0.17 * dst.max()] = [255]
        objects_Contours = []

        count = 0
        area_c = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            area_c += 1
            cnt_3 = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            img = cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
            img = cv2.polylines(img, cnt_3, True, (255, 0, 0), 10)
            cv2.imshow("101010101", img)

        cv2.imshow("Image", img)
        if key == ord('q'):
            break
    cv2.destroyAllWindows()

#
#         image_Gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#         image_Blur = cv.GaussianBlur(image_Gray, (5, 5), 1)
#         image_Canny = cv.Canny(image_Blur, threshold1=threshold[0], threshold2=threshold[1])  # 150,100
#
#         # Necessário Para Realizar A Amostragem Na Matriz 4X4 Devido Às img "Gray" e "Canny" Serem A
#         # Preto E Branco, Logo A "Variável" Type:= Numpy Array Passou A Ser De Dimensão "2" Pois Perdeu A 3ª Indexação
#         # Que Corresponde Ao "RGB". As Imagems A Cores Em OpenCV São Indexadas Em Arrays Do Tipo Numpy Array Com Dimensão "3" Sendo
#         # A 3ª Indexação O "RGB".
#         image_Canny_To_Color_Dimension = cv.cvtColor(image_Canny, cv.COLOR_GRAY2RGB)
#
#         kernel = np.ones((5, 5))
#         image_dilate = cv.dilate(image_Canny, kernel, iterations=3)  # 3)
#         image_Threshold = cv.erode(image_dilate, kernel, iterations=2)  # 2) Mas Pode Ser 1.
#
#         ####contours, hierarchy = cv.findContours(image_Threshold,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE) # Find The Countours.
#         contours, hierarchy = cv.findContours(image_Threshold, cv.RETR_EXTERNAL,
#                                               cv.CHAIN_APPROX_SIMPLE)  # Find The Countours.
#
#
#             ################################## São Iguais #############################
#             """
#             img_02 = cv.drawContours(img_02, cnt, -1, (0, 0, 255), 7)
#             img_2 = img_10
#             img_2 = cv.polylines(img_2,cnt, True, (0, 0 ,255), 8)
#             """
#             ############################################################################
#
#             if area > 2000:
#                 count += 1
#
#                 cnt_1 = cv.approxPolyDP(cnt, 0.01 * cv.arcLength(cnt, True), True)
#                 print("\nCNT_1:=....", len(cnt_1))
#                 #######objects_Contours.append(cnt_1)
#                 # img_01_03 = cv.drawContours(img_01, contours, -1 ,(0, 255, 0), 2)
#                 # img_02_04 = cv.drawContours(img_02, objects_Contours, -1 ,(0, 255, 0), 2,cv.LINE_AA)
#
#                 # cnt_2 = cnt_1
#                 img_01 = cv.polylines(img_01, cnt_1, True, (255, 0, 0),
#                                       20)  # Quando Aplicado Esta Função Muda o Valor De "cnt_1".
#                 objects_Contours.append(cnt_1)
#                 # img_02 = img_01
#                 # img_02 = cv.drawContours(img_02, objects_Contours, -1, (0, 255, 0), 2, cv.LINE_AA)
#                 img_01 = cv.drawContours(img_01, objects_Contours, -1, (0, 255, 0), 2, cv.LINE_AA)
#
#                 #### Amostragem Dos Pontos Individuais #####
#                 # img_03 = cv.polylines(img_03, cnt_1, True, (255, 0, 0), 9)
#                 print("\nCNT_1:=....", len(cnt_1))
#                 print("\nCNT_1:=....", cnt_1)
#                 print("\nOBJECT_COUNTOURS:=...", len(objects_Contours))
#                 # print("\nOBJECT_COUNTOURS:=...",objects_Contours)
#                 pontos_finalis_1 = cnt_1
#
#                 """
#                 rect = cv.minAreaRect(cnt_1)
#                 (x, y), (w, h), angle = rect
#                 #print("X:=...",x,"Y:=...",y)
#                 # Display rectangle
#                 box = cv.boxPoints(rect)
#                 box = np.int0(box)
#
#                 #####object_Width = w / pixel_To_Cm_Ratio
#                 #####object_Height = h / pixel_To_Cm_Ratio
#                 """
#
#         # Get Width and Height of the Objects by applying the Ratio pixel to cm
#         # object_width = w / pixel_cm_ratio
#         # object_height = h / pixel_cm_ratio
#
#         if pontos_finalis_1.any():
#             print("PONTOS_FINALIS:=...", len(pontos_finalis_1))
#             print("PONTOS_FINALIS:=...", pontos_finalis_1)
#             # c2 = cv.approxPolyDP(pontos_finalis, 0.01*cv.arcLength(cnt_1, True), True,)
#             # print("PONTOS_C2:=...",c2)
#             # img_07 = cv.polylines(img_07, c2, True, (255, 0, 0), 10)
#             img_07 = cv.polylines(img_07, pontos_finalis_1, True, (250, 180, 5), 10)
#
#         """
#         thresh = cv.threshold(image_Blur, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]  # 150,255
#         contours_1, hierarchy = cv.findContours(thresh,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
#
#
#         for cnt in contours_1:
#                     area = cv.contourArea(cnt)
#                     if area > 1000:
#                         #for i in range()
#                         cnt_1 = cv.approxPolyDP(cnt, 0.01*cv.arcLength(cnt, True), True,)
#                         objects_Contours.append(cnt_1)
#                         img_04 = cv.drawContours(img_04, cnt_1, -1 ,(0, 255, 0), 2)
#                         img_05 = cv.drawContours(img_05, objects_Contours, -1 ,(0, 255, 0), 2)
#                         img_06 = cv.polylines(img_06, cnt_1, True, (255, 0, 0), 9)
#         """
#
#         # Necessário Para Realizar A Amostragem Na Matriz 4X4 Devido Às img "Gray" e "Canny" Serem A
#         # Preto E Branco, Logo A "Variável" Type:= Numpy Array Passou A Ser De Dimensão "2" Pois Perdeu A 3ª Indexação
#         # Que Corresponde Ao "RGB". As Imagems A Cores Em OpenCV São Indexadas Em Arrays Do Tipo Numpy Array Com Dimensão "3" Sendo
#         # A 3ª Indexação O "RGB".
#
#         # image_Canny_To_Color_Dimension = cv.cvtColor(image_Canny, cv.COLOR_GRAY2RGB)
#
#         image_Resizing_0 = cv2.resize(image, dsize=(0, 0), fx=F_XY[0], fy=F_XY[1])  # fy = 0.5)
#         image_Resizing_1 = cv2.resize(image_Canny_To_Color_Dimension, dsize=(0, 0), fx=F_XY[0], fy=F_XY[1])
#         image_Resizing_2 = cv2.resize(img_01, dsize=(0, 0), fx=F_XY[0], fy=F_XY[1])
#         image_Resizing_3 = cv2.resize(img_02, dsize=(0, 0), fx=F_XY[0], fy=F_XY[1])
#
#         """
#         image_Resizing_0 = cv2.resize(image, dsize = (0,0), fx = 0.5, fy = 0.5)
#         image_Resizing_1 = cv2.resize(image_Canny_To_Color_Dimension, dsize = (0,0), fx = 0.5, fy = 0.5)
#         image_Resizing_2 = cv2.resize(img_01, dsize = (0,0), fx = 0.5, fy = 0.5)
#         image_Resizing_3 = cv2.resize(img_02, dsize = (0,0), fx = 0.5, fy = 0.5)
#         """
#
#         image_Resizing_4 = cv2.resize(img_04, dsize=(0, 0), fx=0.5, fy=0.5)
#         image_Resizing_5 = cv2.resize(img_05, dsize=(0, 0), fx=0.5, fy=0.5)
#         image_Resizing_6 = cv2.resize(img_06, dsize=(0, 0), fx=0.5, fy=0.5)
#         image_Resizing_7 = cv2.resize(img_07, dsize=(0, 0), fx=0.5, fy=0.5)
#         image_Resizing_8 = cv2.resize(img_09, dsize=(0, 0), fx=0.5, fy=0.5)
#
#         image_Final = concatenacao_vh([[image_Resizing_0, image_Resizing_1],
#                                        [image_Resizing_2, image_Resizing_3],
#                                        ]
#                                       )
#         # show the output image
#         cv2.imshow("Matriz", image_Final)
#
#         cv.imshow("GRAY", image_Gray)
#         cv.imshow("BLUR", image_Blur)
#         cv.imshow("CANNY", image_Canny)
#         cv.imshow("DILATE", image_dilate)
#         cv.imshow("THRESHOLD", image_Threshold)
#         cv.imshow("IMAGE 01", img_01)
#         cv.imshow("IMAGE 02", img_02)
#         cv.imshow("IMAGE 03", img_03)
#         cv.imshow("IMAGE 04", img_04)
#         cv.imshow("IMAGE 05", img_05)
#         cv.imshow("IMAGE 06", img_06)
#         cv.imshow("IMAGE 07", img_07)
#         # cv.imshow("IMAGE 002",img_2)
#
#         print("\n\nAREA_C:=...", area_c)
#         print("\n\nAREA:=...", area)
#         print("\n\nCOUNT:=...", count)
#
#         plt.imshow(image)
#         # print("OBJECT_COUNTOURS:=...",objects_Contours)
#         ###########################################array_Pontos_Finallis_2D = np.reshape(pontos_finalis_1,(6,2))
#         print("PONTOS_FINALIS_1:=...", pontos_finalis_1, "SHAPE:=...", np.shape(pontos_finalis_1), ",SIZE:=...",
#               np.size(pontos_finalis_1), "DIMENSÃO:=...", np.ndim(pontos_finalis_1))
#         #########################################print("PONTOS_FINALLIS_2D:=...",array_Pontos_Finallis_2D, "SHAPE:=...",np.shape(array_Pontos_Finallis_2D), ",SIZE:=...", np.size(array_Pontos_Finallis_2D), "DIMENSÃO:=...", np.ndim(array_Pontos_Finallis_2D) )
#         print("THRESHOLD:=...", threshold)
#         print("F_XY:=...", F_XY)
#         print("SWITCH:=...", switch_Variable)
#
#         cv.imwrite("/Sobrantes/Photo_01.jpg", img_01)
#
#         cv.waitKey(0)
