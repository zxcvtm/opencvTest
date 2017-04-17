package com.example.juan.opencvtest;

import android.Manifest;
import android.annotation.TargetApi;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.SurfaceView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.InstallCallbackInterface;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.*;   // VideoCapture

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener {

    private static final String TAG="mainActivity";
    private JavaCameraView  javaCameraView;
    private Mat mRbga;


    Mat outerBox = new Mat();
    Mat diff_frame = null;
    Mat tempon_frame = null;
    private static Mat imag;
    ArrayList<Rect> array = new ArrayList<Rect>();
    private int i=0;
    private int count =0;

    private BaseLoaderCallback mLoaderCallBack = new BaseLoaderCallback(this) {
        @TargetApi(Build.VERSION_CODES.HONEYCOMB)
        @Override
        public void onManagerConnected(int status) {

            switch (status){
                case BaseLoaderCallback.SUCCESS:{
                    javaCameraView.enableView();
                    break;
                }
                default:{
                    super.onManagerConnected(status);
                    break;
                }
            }
        }

        @Override
        public void onPackageInstall(int operation, InstallCallbackInterface callback) {
            super.onPackageInstall(operation, callback);
        }
    };



    static{
        if(!OpenCVLoader.initDebug()){
            Log.d(TAG, "opencv not loaded");
        }else{
            Log.d(TAG, "opencv loaded");
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Here, thisActivity is the current activity
        if (ContextCompat.checkSelfPermission(this,
                Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {

            // Should we show an explanation?
            if (ActivityCompat.shouldShowRequestPermissionRationale(this,
                    Manifest.permission.CAMERA)) {

                // Show an expanation to the user *asynchronously* -- don't block
                // this thread waiting for the user's response! After the user
                // sees the explanation, try again to request the permission.

            } else {

                // No explanation needed, we can request the permission.

                ActivityCompat.requestPermissions(this,
                        new String[]{Manifest.permission.CAMERA},
                        10);

                // MY_PERMISSIONS_REQUEST_READ_CONTACTS is an
                // app-defined int constant. The callback method gets the
                // result of the request.
            }
        }

        javaCameraView= (JavaCameraView)this.findViewById(R.id.camara);
        javaCameraView.setVisibility(SurfaceView.VISIBLE);
        javaCameraView.setCvCameraViewListener(this);

    }
    @Override
    public void onRequestPermissionsResult(int requestCode,
                                           String permissions[], int[] grantResults) {
        switch (requestCode) {
            case 10: {
                if (grantResults.length > 0
                        && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    Log.d(TAG, "Permisos aceptados");
                } else {
                    Log.d(TAG, "Permisos rechazados");
                }
                return;
            }

        }
    }
    @Override
    protected void onPause(){
        super.onPause();
        if(javaCameraView!=null){
            javaCameraView.disableView();
        }
    }
    @Override
    protected void onDestroy(){
        super.onDestroy();
        if(javaCameraView!=null){
            javaCameraView.disableView();
        }
    }
    @Override
    protected void onResume(){
        super.onResume();
        if(!OpenCVLoader.initDebug()){
            Log.d(TAG, "opencv not loaded");
            mLoaderCallBack.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }else{
            Log.d(TAG, "opencv loaded");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_9,this, mLoaderCallBack);
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        mRbga = new Mat(height,width, CvType.CV_8UC4);
    }

    @Override
    public void onCameraViewStopped() {

    }

    @Override
    public Mat onCameraFrame(Mat inputFrame) {

        mRbga = inputFrame;
        count++;
        if((count%1)==0) {
            //Imgproc.resize(inputFrame, inputFrame, inputFrame.size());
            imag = inputFrame.clone();
            Mat outerBox = new Mat(inputFrame.size(), CvType.CV_8UC1);
            Imgproc.cvtColor(inputFrame, outerBox, Imgproc.COLOR_BGR2GRAY);
            Imgproc.GaussianBlur(outerBox, outerBox, new Size(3, 3), 0);
            if (i == 0) {
                diff_frame = new Mat(outerBox.size(), CvType.CV_8UC1);
                tempon_frame = new Mat(outerBox.size(), CvType.CV_8UC1);
                diff_frame = outerBox.clone();
            }
            if (i == 1) {
                Core.subtract(outerBox, tempon_frame, diff_frame);
                Imgproc.adaptiveThreshold(diff_frame, diff_frame, 255,
                        Imgproc.ADAPTIVE_THRESH_MEAN_C,
                        Imgproc.THRESH_BINARY_INV, 5, 2);
                array = detection_contours(diff_frame);
                if (array.size() > 0) {
                    Iterator<Rect> it2 = array.iterator();
                    while (it2.hasNext()) {
                        Rect obj = it2.next();
                        Imgproc.rectangle(imag, obj.br(), obj.tl(),
                                new Scalar(0, 255, 0), 1);
                    }

                }
            }

            i = 1;
            tempon_frame = outerBox.clone();
        }else{
            return mRbga;
        }
        return imag;
        
    }

    public static ArrayList<Rect> detection_contours(Mat outmat) {
        Mat v = new Mat();
        Mat vv = outmat.clone();
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Imgproc.findContours(vv, contours, v, Imgproc.RETR_LIST,
                Imgproc.CHAIN_APPROX_SIMPLE);

        double maxArea = 1000;
        int maxAreaIdx = -1;
        Rect r = null;
        ArrayList<Rect> rect_array = new ArrayList<Rect>();

        for (int idx = 0; idx < contours.size(); idx++) {
            Mat contour = contours.get(idx);
            double contourarea = Imgproc.contourArea(contour);
            if (contourarea > maxArea) {
            // maxArea = contourarea;
            maxAreaIdx = idx;
            r = Imgproc.boundingRect(contours.get(maxAreaIdx));
            rect_array.add(r);
            Imgproc.drawContours(imag, contours, maxAreaIdx, new Scalar(0,0, 255));
        }

        }

        v.release();

        return rect_array;

    }

}