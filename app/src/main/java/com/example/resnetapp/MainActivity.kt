package com.example.resnetapp

import android.graphics.Bitmap
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView

import android.content.Intent

import android.provider.MediaStore
import android.util.Log
import com.example.resnetapp.ml.MobilenetV110224Quant
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.TransformToGrayscaleOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer


class MainActivity : AppCompatActivity() {

    private val TAG = "chevin_2222"
    lateinit var selectBtn: Button
    lateinit var predBtn: Button
    lateinit var resView: TextView
    lateinit var imageView: ImageView
    lateinit var bitmap: Bitmap
    val one = 1
    val two = 2

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        selectBtn = findViewById(R.id.selectBtn)
        predBtn = findViewById(R.id.predictBtn)
        resView = findViewById(R.id.resView)
        imageView = findViewById(R.id.imageView)


        var labels = application.assets.open("labels.txt").bufferedReader().readLines()


        // image processor to use
        var imageProcessor = ImageProcessor.Builder()
         //   .add(NormalizeOp(0.0f, 255.0f))
         //   .add(TransformToGrayscaleOp())
            .add(ResizeOp(224,224, ResizeOp.ResizeMethod.BILINEAR)).build()


        var result = one + two
        Log.d(TAG, "$result")

        selectBtn.setOnClickListener{

            val intent = Intent()
            intent.action = Intent.ACTION_GET_CONTENT
            intent.type = "image/*"
            startActivityForResult(intent, 100)
        }

        predBtn.setOnClickListener {

            var tensorImage = TensorImage(DataType.UINT8)
            tensorImage.load(bitmap)

            tensorImage = imageProcessor.process(tensorImage)


            val model = MobilenetV110224Quant.newInstance(this)

            // Creates inputs for reference.
            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.UINT8)
            inputFeature0.loadBuffer(tensorImage.buffer)

            // Runs model inference and gets result.
            val outputs = model.process(inputFeature0)
            val outputFeature0 = outputs.outputFeature0AsTensorBuffer.floatArray

            var maxIdx = 0
            outputFeature0.forEachIndexed { index, fl ->
                if(outputFeature0[maxIdx] < fl){
                    maxIdx = index
                }
            }

            resView.setText(labels[maxIdx])


            // Releases model resources if no longer used.
            model.close()
        }

    }

    override fun onActivityResult(requestCode: Int, resultCode: Int,data:Intent?){
        super.onActivityResult(requestCode, resultCode, data)
        Log.d(TAG, "${resultCode} ")
        if(requestCode == 100){

            var uri = data?.data
            bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)
            imageView.setImageBitmap(bitmap)

        }
    }
}