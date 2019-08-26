package com.example.acam;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.FileProvider;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;
import java.util.Arrays;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;

public class MainActivity extends AppCompatActivity {

    private Button btnCamera;
    private Button btnCamera1;
    private ImageView capturedImage;
    private Bitmap bp;
    private EditText editText;
    private TensorFlowInferenceInterface inferenceInterface;
    private String dat[]={"eraser","pen","sharpner","pencil"};


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

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

        editText=(EditText)findViewById(R.id.editText);


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

            String label=editText.getText().toString()+".jpeg";

            String path=Environment.getExternalStoragePublicDirectory(
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

        // return prediction
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
        float mmax=0;
        int mmaxi=-1;
        for(int i=0;i<4;i++)
        {
            if(result[i]>mmax)
            {
                mmax=result[i];
                mmaxi=i;
            }
        }
        editText.setText(dat[mmaxi]);

    }
}
