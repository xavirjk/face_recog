#include <iostream>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

Mat initialImageProcessing(Mat frame);
void detectTheEyes(Mat &face, Point &leftEye, Point &rightEye, CascadeClassifier &eyesCascade1, CascadeClassifier &eyesCascade2);
void detectObj(const Mat &img, CascadeClassifier &cascade, Rect &obj, int scaledw);
void equilizefaceHalves(Mat &faceimg);

CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade1, eyes_cascade2;
string face_cascadeName = "../haarcascades/haarcascade_frontalface_alt2.xml";
string eyes_cascadeName1 = "../haarcascades/haarcascade_eye.xml";
string eyes_cascadeName2 = "../haarcascades/haarcascade_eye_tree_eyeglasses.xml";

int main(int argc, char *argv[])
{
    try
    {
        face_cascade.load(face_cascadeName);
        eyes_cascade1.load(eyes_cascadeName1);
        eyes_cascade2.load(eyes_cascadeName2);
    }
    catch (Exception &e)
    {
        std::cerr << e.what() << '\n';
        if (face_cascade.empty())
        {
            cerr << " Error!! Could not load faceDetector." << endl;
            cerr << face_cascadeName << endl;
            exit(-1);
        }
    }

    VideoCapture capture(0);
    if (!capture.isOpened())
    {
        cerr << " Error!! could not open Webcam" << endl;
        exit(-1);
    }

    for (;;)
    {
        Mat frame;

        capture >> frame;

        if (frame.empty())
        {
            cout << "No frame" << endl;
            break;
        }
        Mat img;
        const int DetectionWidth = 320;
        float scale = frame.cols / (float)DetectionWidth;

        if (frame.cols > DetectionWidth)
        {
            int scaledHeight = cvRound(frame.rows / scale);
            resize(frame, img, Size(DetectionWidth, scaledHeight));
        }
        else
            img = frame;
        Mat equilizedimg = initialImageProcessing(img);
        vector<Rect> faces;
        face_cascade.detectMultiScale(equilizedimg, faces, 1.1f, 4, CASCADE_FIND_BIGGEST_OBJECT, Size(20, 20));

        if (frame.cols > DetectionWidth)
        {
            for (int i = 0; i < faces.size(); i++)
            {
                int scaled_x, scaled_y, scaled_h, scaled_w;

                Point Pt1(faces[i].x, faces[i].y);
                Point Pt2(faces[i].height + faces[i].x, faces[i].width + faces[i].y);

                scaled_x = cvRound(faces[i].x * scale);
                scaled_y = cvRound(faces[i].y * scale);
                scaled_w = cvRound(faces[i].width * scale);
                scaled_h = cvRound(faces[i].height * scale);

                Point Pt3(scaled_x, scaled_y);
                Point Pt4(scaled_h + scaled_x, scaled_w + scaled_y);

                rectangle(img, Pt1, Pt2, Scalar(0, 255, 0), 2, 8, 0);
                rectangle(frame, Pt3, Pt4, Scalar(0, 255, 0), 2, 8, 0);

                Mat faceROI = equilizedimg(faces[i]);
                imshow("faceimg", faceROI);
                vector<Rect> eyes;
                Point leftEye, rightEye;
                //-- In each face, detect eyes
                detectTheEyes(faceROI, leftEye, rightEye, eyes_cascade1, eyes_cascade2);
                if (leftEye.x > 0 && rightEye.x > 0)
                {
                    cout << "both detected" << endl;
                    Point center1(faces[i].x + leftEye.x, faces[i].y + leftEye.y);
                    Point center2(faces[i].x + rightEye.x, faces[i].y + rightEye.y);
                    int radius = 20;
                    circle(img, center1, radius, Scalar(255, 0, 0), 4, 8, 0);
                    circle(img, center2, radius, Scalar(255, 0, 0), 4, 8, 0);
                    //Center b2n the 2 eyes
                    Point2f eyesCenter;
                    eyesCenter.x = (leftEye.x + rightEye.x) * 0.5f;
                    eyesCenter.y = (leftEye.y + rightEye.y) * 0.5f;
                    //Angle b2n the 2 eyes
                    double dy = rightEye.y - leftEye.y;
                    double dx = rightEye.x - leftEye.x;
                    double len = sqrt(dx * dx + dy * dy);
                    //Convert radians to degrees

                    double angle = atan2(dy, dx) * 180.0 / CV_PI;

                    const double DESIRED_LEFT_EYE_X = 0.16;
                    const double DESIRED_LEFT_EYE_Y = 0.14;
                    const double DESIRED_RIGHT_EYE_X = (1.0f - 0.16);
                    //const double DESIRED_RIGHT_EYE_X = (0.8);

                    const int DESIRED_FACE_WIDTH = 70;
                    const int DESIRED_FACE_HEIGHT = 70;

                    double desired_len = (DESIRED_RIGHT_EYE_X - DESIRED_LEFT_EYE_X) * DESIRED_FACE_WIDTH;
                    double scale = desired_len / len;

                    Mat rotated_mat = getRotationMatrix2D(eyesCenter, angle, scale);

                    double ex = DESIRED_FACE_WIDTH * 0.5f - eyesCenter.x;
                    double ey = DESIRED_FACE_HEIGHT * DESIRED_LEFT_EYE_Y - eyesCenter.y;

                    rotated_mat.at<double>(0, 2) += ex;
                    rotated_mat.at<double>(1, 2) += ey;

                    Mat warped = Mat(DESIRED_FACE_HEIGHT, DESIRED_FACE_WIDTH, CV_8U, Scalar(128));
                    imshow("Face img b4 pre", faceROI);
                    warpAffine(faceROI, warped, rotated_mat, warped.size());
                    imshow("Face img afta pre", warped);
                    equilizefaceHalves(warped);
                    imshow("fame img equilized halves", warped);

                    Mat filtered = Mat(warped.size(), CV_8U);
                    bilateralFilter(warped, filtered, 0, 20.0, 2.0);
                    imshow("filtered", filtered);

                    Mat mask = Mat(warped.size(), CV_8U, Scalar(0));
                    Point faceCenter = Point(cvRound(DESIRED_FACE_WIDTH * 0.5), cvRound(DESIRED_FACE_HEIGHT * 0.4));
                    Size size = Size(cvRound(DESIRED_FACE_WIDTH * 0.5), cvRound(DESIRED_FACE_HEIGHT * 0.8));
                    ellipse(mask, faceCenter, size, 0, 0, 360, Scalar(255), FILLED);

                    Mat dest = Mat(warped.size(), CV_8U, Scalar(128));
                    filtered.copyTo(dest, mask);
                    imshow("masked", dest);
                }
            }
        }
        imshow("Equilizedimage", equilizedimg);
        imshow("original", img);
        imshow("ORfarme", frame);

        char c = (char)waitKey(25);

        if (c == 27)
            break;
    }
    return 0;
}

Mat initialImageProcessing(Mat frame)
{
    Mat gray;
    if (frame.channels() == 3)
        cvtColor(frame, gray, COLOR_BGR2GRAY);
    else if (frame.channels() == 4)
        cvtColor(frame, gray, COLOR_BGRA2GRAY);
    else
        gray = frame;
    Mat equilizedImage;
    equalizeHist(gray, equilizedImage);
    return equilizedImage;
}
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

void equilizefaceHalves(Mat &faceImg)
{
    int width = faceImg.cols;
    int height = faceImg.rows;

    //equalize the whole face.
    Mat wholeFace;
    equalizeHist(faceImg, wholeFace);

    //Equalize left and the right half of the face separately.
    int middle = width / 2;
    Mat leftSide = faceImg(Rect(0, 0, middle, height));
    Mat rightSide = faceImg(Rect(middle, 0, width - middle, height));
    equalizeHist(leftSide, leftSide);
    equalizeHist(rightSide, rightSide);

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int v;
            if (x < width / 4)
            {
                // Left 25%
                v = leftSide.at<uchar>(y, x);
            }
            else if (x < width * 2 / 4)
            {
                // Mid-left 25%
                int lv = leftSide.at<uchar>(y, x);
                int wv = wholeFace.at<uchar>(y, x);
                // Blend more of the whole face as it moves further right along the face.
                float f = (x - width * 1 / 4) / (float)(width * 0.25f);
                v = cvRound((1.0f - f) * lv + (f)*wv);
            }
            else if (x < width * 3 / 4)
            {
                // Mid-right 25%
                int rv = rightSide.at<uchar>(y, x - middle);
                int wv = wholeFace.at<uchar>(y, x);
                // Blend more of the right-side face as it moves further right along the face.
                float f = (x - width * 2 / 4) / (float)(width * 0.25f);
                v = cvRound((1.0f - f) * wv + (f)*rv);
            }
            else
            {
                // Right 25%
                v = rightSide.at<uchar>(y, x - middle);
            }
            faceImg.at<uchar>(y, x) = v;
        }
    }
}