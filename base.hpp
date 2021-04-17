#ifndef BASE_H
#define BASE_H

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <obj-detect/obj-detect.hpp>
#include <img-process/img-process.hpp>
#include <recognition/recognition.hpp>

namespace faceRecog
{
    namespace init
    {
        class faceRecognizer
        {
        private:
            CascadeClassifier face_cascade;
            CascadeClassifier eyes_cascade1, eyes_cascade2;
            String face_cascadeName = "../../haarcascades/haarcascade_frontalface_alt2.xml";
            String eyes_cascadeName1 = "../../haarcascades/haarcascade_eye.xml";
            String eyes_cascadeName2 = "../../haarcascades/haarcascade_eye_tree_eyeglasses.xml";

            VideoCapture capture;
            vector<Rect> faces;
            Rect faceRect;
            Point leftEye, rightEye;
            int scaledHeight;
            const int DetectionWidth = 320;
            float scale;

            Mat previous_preprocessed_face;
            Mat preProcessedFace;
            vector<Mat> preprocessedFaces;
            Ptr<EigenFaceRecognizer> model;
            int _identity = 0;
            vector<int> labels;
            double pre_time = 0;

            enum MODES
            {
                STARTUP_MODE,
                DETECTION_MODE,
                FACES_COLLECT_MODE,
                TRAINING_MODE,
                RECOG_MODE
            };

            MODES r_mode = STARTUP_MODE;
            const char *MODES_NAMES[5] = {"startup", "detection", "collect Faces", "train model", "recognize face"};
            const float UNKNOWN_PERSON_THRESHOLD = 0.35f;

            void collectFaces(Mat &frame);
            void trainModel();
            void recognizeface();

        public:
            faceRecognizer();
            ~faceRecognizer();
            void initWebcam();
            void recognizer();
            Mat scaleImage(Mat frame);
            Mat gray_ScaledImage(Mat frame);
            void drawFaces(Mat &frame, Mat &img, int i);
            void drawEyes(Mat &img, int i);
            void setMode(int mode);
        };
    }
}

#endif