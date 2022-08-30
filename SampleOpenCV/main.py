import cv2
import numpy


def drawKP():
    imgTrainPath = 'F:/poi_arrow.png'
    imgTrain = cv2.imread(imgTrainPath)  # 加载图片
    gray = cv2.cvtColor(imgTrain, cv2.COLOR_BGR2GRAY)  # 转换颜色空间

    surf = cv2.xfeatures2d.SURF_create()  # 创建SURF对象，这里可以传入参数设置阈值

    keypoints, descriptor = surf.detectAndCompute(gray, None)  # keypoints 特征点位；descriptor，描述符

    # 把特征点画在原图片上
    img = cv2.drawKeypoints(image=imgTrain,
                            outImage=imgTrain,
                            keypoints=keypoints,
                            flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT,
                            color=(51, 163, 236))

    cv2.imshow('sift_keypoints', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 对比图片并描点
def compareDemo():
    # 参考SIFT feature_matching point coordinates https://stackoverflow.com/questions/46607647/sift-feature-matching-point-coordinates
    # https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
    # https://docs.opencv.org/4.x/db/d27/tutorial_py_table_of_contents_feature2d.html

    imgQueryPath = 'F:/android.png'  # 源图 queryImage
    imgTrainPath = 'F:/android-small.png'  # 被检索目标 trainImage
    imgQuery = cv2.imread(imgQueryPath)  # 加载图片
    imgTrain = cv2.imread(imgTrainPath)  # 加载图片

    # Create SIFT object
    sift = cv2.xfeatures2d.SIFT_create()
    # Create SURF object
    surf = cv2.xfeatures2d.SURF_create()  # 创建SURF对象，这里可以传入参数设置阈值
    # Initiate ORB detector
    orb = cv2.ORB_create()

    # surf.setUpright(True)       # 关闭方向检测，速度更快
    # print(surf.getUpright())

    # Find keypoints and descriptors directly
    kpQuery, desQuery = surf.detectAndCompute(imgQuery, None)
    kpTrain, desTrain = surf.detectAndCompute(imgTrain, None)

    # create BFMatcher object for SIFT、SURF detector
    bfMatcherL2 = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    # create BFMatcher object for ORB detector
    bfMatcherHamming = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bfMatcherL2.match(desQuery, desTrain)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw first 10 matches.
    img3 = cv2.drawMatches(imgQuery, kpQuery, imgTrain, kpTrain, matches[:10], None,
                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('sift_keypoints', img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def getPointAndAngel():
    # https://blog.csdn.net/qq_26671711/article/details/62891143
    # https://stackoverflow.com/questions/37583375/how-to-use-opencv-featuredetecter-on-tiny-images  似乎opencv无法检测很细小的图片

    imgQueryPath = 'F:/android-20.png'  # 源图 queryImage
    # imgQueryPath = 'F:/000.png'
    imgTrainPath = 'F:/android-30.png'  # 被检索目标 trainImage
    # imgTrainPath = 'F:/aaa.png'
    imgQuery = cv2.imread(imgQueryPath)  # 加载图片
    imgTrain = cv2.imread(imgTrainPath)  # 加载图片

    # Create SIFT object
    sift = cv2.xfeatures2d.SIFT_create()
    # Create SURF object
    surf = cv2.xfeatures2d.SURF_create()  # 创建SURF对象，这里可以传入参数设置阈值
    # Initiate ORB detector
    orb = cv2.ORB_create()

    # surf.setUpright(True)       # 关闭方向检测，速度更快
    # print(surf.getUpright())

    # Find keypoints and descriptors directly
    kpQuery, desQuery = surf.detectAndCompute(imgQuery, None)
    kpTrain, desTrain = surf.detectAndCompute(imgTrain, None)

    # create BFMatcher object for SIFT、SURF detector
    bfMatcherL2 = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)      # knnMatch需要把crossCheck设置为False
    # create BFMatcher object for ORB detector
    bfMatcherHamming = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # Match descriptors.
    matches = bfMatcherL2.knnMatch(desQuery, desTrain, 2)       # 匹配的k参数数量，代表你会找到几个匹配结果，也就是下面的m1，m2
    matchesMask = [[0, 0] for i in range(len(matches))]  # 列表生成式，这其实是生成了一个长度为len(matches)，每个值都是[0,0]的空[]
    for i, (m1, m2) in enumerate(matches):      # 这里的m1和m2和knnMatch中设定的k有关，k=2就会有两个匹配结果
        if m1.distance < 0.7 * m2.distance:     # 描述符之间的距离差对比，实际看在0.4-0.8之间，0.8以上有很多错误匹配，0.4以下几乎匹配不上
            matchesMask[i] = [1, 0]
            # Notice: How to get the index
            pt1 = kpQuery[m1.queryIdx].pt       # query图上的点坐标
            pt2 = kpTrain[m2.trainIdx].pt       # train图上的点坐标
            print(i, pt1, pt2)
            angle1 = kpQuery[m1.queryIdx].angle
            angle2 = kpTrain[m2.trainIdx].angle
            print(i, angle1, angle2)

    # Draw match in blue, error in red
    draw_params = dict(matchColor=(255, 0, 0),
                       singlePointColor=(0, 0, 255),
                       matchesMask=matchesMask,
                       flags=0)

    res = cv2.drawMatchesKnn(imgQuery, kpQuery, imgTrain, kpTrain, matches, None, **draw_params)
    cv2.imshow("Result", res)
    cv2.waitKey()
    cv2.destroyAllWindows()


print("Hello, OpenCV!")
# drawKP()
# compareDemo()
getPointAndAngel()
