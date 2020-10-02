package com.prskid1000.guesspic;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.os.Environment;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.File;
import java.io.FileOutputStream;

public class MainActivity extends AppCompatActivity {

    private Button btnCamera;
    private Button btnCamera1;
    private ImageView capturedImage;
    private Bitmap bp;
    private EditText editText1;
    private EditText editText2;
    private EditText editText3;
    private EditText editText4;
    private TensorFlowInferenceInterface inferenceInterface;
    private String dat[]={"eraser","pen","sharpner","pencil"};


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        getSupportActionBar().hide();

        btnCamera = (Button) findViewById(R.id.btnCamera);

        capturedImage= (ImageView) findViewById(R.id.capturedImage);

        btnCamera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                openCamera();
            }
        });

        btnCamera1 = (Button) findViewById(R.id.btnCamera2);

        btnCamera1.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                getResult();
            }
        });

        editText1=(EditText)findViewById(R.id.editText6);
        editText2=(EditText)findViewById(R.id.editText7);
        editText3=(EditText)findViewById(R.id.editText8);
        editText4=(EditText)findViewById(R.id.editText9);


    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        // TODO Auto-generated method stub
        super.onActivityResult(requestCode, resultCode, data);
        if(resultCode == RESULT_OK) {

            bp = (Bitmap) data.getExtras().get("data");
            bp=Bitmap.createScaledBitmap(bp,100,100,true);
            bp=bp.copy(Bitmap.Config.ARGB_8888, false);
            capturedImage.setImageBitmap(bp);

            String label="data.jpeg";

            String path= Environment.getExternalStoragePublicDirectory(
                    Environment.DIRECTORY_PICTURES).toString();
            File file = new File(path,label);


            try {
                FileOutputStream out = new FileOutputStream(file);
                bp.compress(Bitmap.CompressFormat.JPEG, 100, out);
                out.flush();
                out.close();
            }
            catch (Exception e) {
                e.printStackTrace();
            }
        }
    }


    private float[] predict(float[] input){
        // model has only 1 output neuron
        float output[] = new float[4];

        inferenceInterface.feed("conv2d_1_input", input,1,100,100,3);
        inferenceInterface.run(new String[]{"activation_5/Sigmoid"});
        inferenceInterface.fetch("activation_5/Sigmoid", output);

        return output;
    }

    private void openCamera() {
        Intent intent = new Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE);
        startActivityForResult(intent, 0);
    }



    private void getResult()
    {
        inferenceInterface = new TensorFlowInferenceInterface(getAssets(), "tensorflow_lite_trial.pb");

        float[] input=new float[100*100*3];

        int[] intValues = new int[100*100];

        bp.getPixels(intValues,0,100,0,0,100,100);

        for (int i = 0; i < intValues.length; ++i)
        {
            int val = intValues[i];
            input[i * 3 + 0] = ((val & 0xFF) - 104);
            input[i * 3 + 1] = (((val >> 8) & 0xFF) - 117);
            input[i * 3 + 2] = (((val >> 16) & 0xFF) - 123);
        }

        float result[]= predict(input);

        editText1.setText(Float.toString(result[0]));
        editText2.setText(Float.toString(result[3]));
        editText3.setText(Float.toString(result[1]));
        editText4.setText(Float.toString(result[2]));

    }
}