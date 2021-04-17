#include <obj-detect/obj-detect.hpp>

void detectTheEyes(Mat &face, Point &leftEye, Point &rightEye, CascadeClassifier &eyesCascade1, CascadeClassifier &eyesCascade2)
{

    const float EYES_SX = 0.16f;
    const float EYES_SY = 0.26f;
    const float EYES_SW = 0.30f;
    const float EYES_SH = 0.28f;

    int LEFTX = cvRound(face.cols * EYES_SX);
    int TOPY = cvRound(face.rows * EYES_SY);
    int WIDTHX = cvRound(face.cols * EYES_SW);
    int HEIGHTY = cvRound(face.rows * EYES_SH);
    int RIGHTX = cvRound(face.cols * (1.0 - EYES_SX - EYES_SW)) + 5;

    Mat topLeftOfFace = face(Rect(LEFTX, TOPY, WIDTHX, HEIGHTY));
    Mat topRightofFace = face(Rect(RIGHTX, TOPY, WIDTHX, HEIGHTY));
    imshow("top l o Face", topLeftOfFace);
    imshow("top r o face", topRightofFace);
    Rect eyesRect1, eyesRect2;

    detectObj(topLeftOfFace, eyesCascade1, eyesRect1, topLeftOfFace.cols);
    detectObj(topRightofFace, eyesCascade1, eyesRect2, topRightofFace.cols);

    if (eyesRect1.width > 0)
    {
        eyesRect1.x += LEFTX;
        eyesRect1.y += TOPY;
        leftEye = Point(eyesRect1.x + eyesRect1.width / 2, eyesRect1.y + eyesRect1.height / 2);
    }
    else
    {
        leftEye = Point(-1, -1);
    }

    if (eyesRect2.width > 0)
    {
        eyesRect2.x += RIGHTX;
        eyesRect2.y += TOPY;
        rightEye = Point(eyesRect2.x + eyesRect2.width / 2, eyesRect2.y + eyesRect2.height / 2);
    }
    else
    {
        rightEye = Point(-1, -1);
    }
}

void detectObj(const Mat &img, CascadeClassifier &cascade, Rect &obj, int scaledw)
{
    int flags = CASCADE_FIND_BIGGEST_OBJECT;
    Size minfeaturesSize = Size(20, 20);
    float searchScaleFactor = 1.1f;
    int min_neigh = 4;

    vector<Rect> objects;

    cascade.detectMultiScale(img, objects, searchScaleFactor, min_neigh, flags, minfeaturesSize);

    if (objects.size() > 0)
    {
        obj = (Rect)objects.at(0);
    }
    else
    {
        obj = Rect(-1, -1, -1, -1);
    }
}