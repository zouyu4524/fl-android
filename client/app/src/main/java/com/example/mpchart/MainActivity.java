package com.example.mpchart;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.Manifest;
import android.app.Activity;
import android.app.AlertDialog;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.Resources;
import android.graphics.Color;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Environment;
import android.provider.Settings;
import android.service.autofill.Dataset;
import android.view.View;
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.ProgressBar;
import android.widget.Spinner;
import android.widget.TextView;

import com.github.mikephil.charting.charts.LineChart;
import com.github.mikephil.charting.components.Legend;
import com.github.mikephil.charting.components.XAxis;
import com.github.mikephil.charting.components.YAxis;
import com.github.mikephil.charting.data.Entry;
import com.github.mikephil.charting.data.LineData;
import com.github.mikephil.charting.data.LineDataSet;
import com.github.mikephil.charting.interfaces.datasets.ILineDataSet;
import com.github.mikephil.charting.utils.EntryXComparator;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import com.example.mpchart.TrainModel;

import org.datavec.api.transform.filter.InvalidNumColumns;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;
import org.w3c.dom.Text;

public class MainActivity extends AppCompatActivity implements SensorEventListener,
        AdapterView.OnItemSelectedListener, ActivityCompat.OnRequestPermissionsResultCallback {

    // read and write permissions
    private static final int REQUEST_EXTERNAL_STORAGE = 1;
    private static String[] PERMISSIONS_STORAGE = {
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE
    };

    public static void verifyStoragePermission(Activity activity) {
        // Get permission status
        int permission = ActivityCompat.checkSelfPermission(activity, Manifest.permission.WRITE_EXTERNAL_STORAGE);
        if (permission != PackageManager.PERMISSION_GRANTED) {
            // We don't have permission we request it
            ActivityCompat.requestPermissions(
                    activity,
                    PERMISSIONS_STORAGE,
                    REQUEST_EXTERNAL_STORAGE
            );
        }
    }

    // sensor related
    private static SensorManager mSensorManager;
    private static Sensor mAccelerator;

    private LineChart [] mCharts = new LineChart[3];
    private static String currentActivity = "Jogging";
    private ILineDataSet [] dataSets = new ILineDataSet[3];

    private Thread thread;
    private boolean plotData = true;
    private static final int mCounts = 180;
    private static final int [] sampleShape = {1, 270};
    private static final int [] labelShape = {1, 6};

    // training related
    private static boolean isTraining = false;
    private static boolean isLoading = false;
    private static boolean isLabelling = false;

//    private static DataSet dataSet;

    private void addEntry(SensorEvent event, LineChart mChart, int index) {
        LineData data = mChart.getData();

        if (data != null) {
//            ILineDataSet set = data.getDataSetByIndex(0);
            dataSets[index] = data.getDataSetByIndex(0);
            if (dataSets[index] == null) {
                dataSets[index] = createSet(index);
                data.addDataSet(dataSets[index]);
            }
            if (dataSets[index].getEntryCount() >= mCounts) {
                dataSets[index].removeFirst();
            }
            data.addEntry(new Entry(event.timestamp, event.values[index]), 0);

            data.notifyDataChanged();
            mChart.notifyDataSetChanged(); // very important
            mChart.moveViewToX(data.getEntryCount());
        }
    }

    private void initDataset(int id) {

        mCharts[id].getDescription().setEnabled(true);
        String descript = "";
        switch (id) {
            case 0:
                descript = "Real Time Accelerator X-axis";
                break;
            case 1:
                descript = "Real Time Accelerator Y-axis";
                break;
            case 2:
                descript = "Real Time Accelerator Z-axis";
                break;
        }
        mCharts[id].getDescription().setText(descript);

        mCharts[id].setTouchEnabled(false);
        mCharts[id].setDragEnabled(false);
        mCharts[id].setScaleEnabled(false);
        mCharts[id].setDrawGridBackground(false);
        mCharts[id].setPinchZoom(false);
        mCharts[id].setBackgroundColor(Color.WHITE);
        mCharts[id].setAutoScaleMinMaxEnabled(false);

        LineData data = new LineData();
        data.setValueTextColor(Color.WHITE);
        mCharts[id].setData(data);


        XAxis xl = mCharts[id].getXAxis();
        xl.setTextColor(Color.WHITE);
        xl.setDrawGridLines(false);
        xl.setAvoidFirstLastClipping(true);
        xl.setEnabled(true);

        YAxis leftAxis = mCharts[id].getAxisLeft();
        leftAxis.setTextColor(Color.WHITE);
        leftAxis.setDrawGridLines(false);
        leftAxis.setAxisMaximum(20f);
        leftAxis.setAxisMinimum(-20f);

        YAxis rightAxis = mCharts[id].getAxisRight();
        rightAxis.setEnabled(false);

        mCharts[id].getAxisLeft().setDrawGridLines(false);
        mCharts[id].getXAxis().setDrawGridLines(false);
        mCharts[id].setDrawBorders(false);
    }

    private LineDataSet createSet(int id) {
        LineDataSet set = new LineDataSet(null, "");
        set.setAxisDependency(YAxis.AxisDependency.LEFT);
        set.setLineWidth(0.5f);

        switch (id) {
            case 0:
                set.setColor(Color.RED);
                break;
            case 1:
                set.setColor(Color.GREEN);
                break;
            case 2:
                set.setColor(Color.BLUE);
                break;
        }

        set.setDrawValues(false);
        set.setDrawCircles(false);
        set.setMode(LineDataSet.Mode.CUBIC_BEZIER);
        set.setCubicIntensity(0.2f);
        return set;
    }

    private void feedMultiple() {

        if (thread != null){
            thread.interrupt();
        }

        thread = new Thread(new Runnable() {

            @Override
            public void run() {
                while (true){
                    plotData = true;
                    try {
                        Thread.sleep(10);
                    } catch (InterruptedException e) {
                        // TODO Auto-generated catch block
                        e.printStackTrace();
                    }
                }
            }
        });

        thread.start();
    }

    // generator training samples from collected sensor data
    public double[] groupSensorData() {
        int num = dataSets[0].getEntryCount();
        if (num < mCounts) {
            // haven't collected enough data to group into one entry
            return null;
        }
        double [] sample = new double[270];
        for (int i = 0; i<dataSets[0].getEntryCount(); i+=2) {
            // sampling by performing i+=2, so as to group 90 reads into one entry to meet
            // the size of the model
            for (int index = 0; index < 3; index++) {
                sample[i/2*3+index] = (double)dataSets[index].getEntryForIndex(i).getY();
            }
        }
        return sample;
    }

    // concat newly generated sample to the training dataset
    // error occurs when load from saved dataset, not clear how to solve it
    // https://gitter.im/deeplearning4j/deeplearning4j/archives/2017/10/18
    // the author states that save and load as binary is much efficient than save and load via csv
    @Deprecated
    public void concatTrainingDataSet(INDArray sensorData, INDArray label) {
        // load dataSet
        DataSet dataSet = new DataSet();
        dataSet.load(TrainModel.locateToSaveDataSet);

        // extract features and labels, respectively
        INDArray sensors = dataSet.getFeatures();
        INDArray labels = dataSet.getLabels();

        // concat newly added sample
        sensors = Nd4j.concat(0, sensors, sensorData);
        labels = Nd4j.concat(0, labels, label);

        // save updated dataSet with newly added sample
        dataSet = new DataSet(sensors, labels);
        dataSet.save(TrainModel.locateToSaveDataSet);
    }

    // Sensor related codes
    @Override
    public final void onSensorChanged(SensorEvent sensorEvent) {
        if(plotData) {
            for (int i = 0; i < 3; i++){
                addEntry(sensorEvent, mCharts[i], i);
            }
            plotData = false;
        }
    }

    @Override
    public final void onAccuracyChanged(Sensor sensor, int accuracy) {

    }

    // Spinner related codes
    @Override
    public void onItemSelected(AdapterView<?> parent, View view,
                               int pos, long id) {
        currentActivity = parent.getItemAtPosition(pos).toString();
    }

    @Override
    public void onNothingSelected(AdapterView<?> parent) {

    }

    // main activity related codes
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        verifyStoragePermission(MainActivity.this);

        mSensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        mAccelerator = mSensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION);

        if(mAccelerator != null) {
            mSensorManager.registerListener(this, mAccelerator, SensorManager.SENSOR_DELAY_GAME);
        }

        // init spinner and set listener
        Spinner activities = findViewById(R.id.spinner);
        activities.setOnItemSelectedListener(this);

        mCharts[0] = findViewById(R.id.chart_x);
        mCharts[1] = findViewById(R.id.chart_y);
        mCharts[2] = findViewById(R.id.chart_z);

        for (int i=0; i<3; i++) {
            initDataset(i);
        }

        feedMultiple();

        TrainModel.id = Settings.Secure.getString(getContentResolver(), Settings.Secure.ANDROID_ID);
        Button button = (Button) findViewById(R.id.button_get_model);
        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                if (isLoading || isTraining || isLabelling) {
                    return;
                }
                isLoading = true;
                AsyncTaskRunner runner = new AsyncTaskLoadModel();
                runner.execute();
                ProgressBar bar = (ProgressBar) findViewById(R.id.progressBar);
                bar.setVisibility(View.VISIBLE);
            }
        });

        Button button_train = (Button) findViewById(R.id.button_train);
        button_train.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (isTraining || isLoading || isLabelling) {
                    return;
                }
                if (TrainModel.model == null) {
                    // model is not loaded yet
                    TextView textView = findViewById(R.id.textView);
                    textView.setText(R.string.warning_load_model);
                    return;
                }
                isTraining = true;
                AsyncTaskRunner runner = new AsyncTaskTrainModel();
                runner.execute();
                ProgressBar bar = (ProgressBar) findViewById(R.id.progressBar);
                bar.setVisibility(View.VISIBLE);
            }
        });

        Button button_label = (Button) findViewById(R.id.button_label);
        button_label.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (isTraining || isLoading || isLabelling) {
                    return;
                }
                isLabelling = true;
                AsyncTaskRunner runner = new AsyncTaskLabelling();
                runner.execute();
                ProgressBar bar = (ProgressBar) findViewById(R.id.progressBar);
                bar.setVisibility(View.VISIBLE);
            }
        });

        Button button_infer = (Button) findViewById(R.id.button_inference);
        button_infer.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (isTraining || isLoading || isLabelling) {
                    return;
                }
                if (TrainModel.model == null) {
                    // model is not loaded yet
                    TextView textView = findViewById(R.id.textView);
                    textView.setText(R.string.warning_load_model);
                    return;
                }
                double [] sample = groupSensorData();
                if (sample == null) {
                    return;
                }
                INDArray sample_to_infer = Nd4j.create(ArrayUtil.flattenDoubleArray(sample), sampleShape);
                INDArray predicted = TrainModel.model.output(sample_to_infer, false);
                INDArray index = predicted.argMax();
                int[] pl = index.toIntVector();
                String predicted_label = "You are " + LocalDataSet.statues[pl[0]].toLowerCase() + ".";
                TextView textView = findViewById(R.id.textView);
                textView.setText(predicted_label);
            }
        });

    }

    @Override
    protected void onPause() {
        super.onPause();

        if (thread != null) {
            thread.interrupt();
        }
        mSensorManager.unregisterListener(this);
    }

    @Override
    protected void onResume() {
        super.onResume();
        mSensorManager.registerListener(this, mAccelerator, SensorManager.SENSOR_DELAY_GAME);
    }

    @Override
    protected void onPostResume() {
        super.onPostResume();
    }

    @Override
    protected void onDestroy() {
        mSensorManager.unregisterListener(MainActivity.this);
        thread.interrupt();
        super.onDestroy();
    }

    private class AsyncTaskRunner extends AsyncTask<Void, Integer, Integer> {

        @Override
        protected void onPreExecute() {
            super.onPreExecute();

            ProgressBar bar = (ProgressBar) findViewById(R.id.progressBar);
            bar.setVisibility(View.INVISIBLE);
        }

        @Override
        protected Integer doInBackground(Void... params) {
            return 0;
        }

        @Override
        protected void onProgressUpdate(Integer... values) {
            super.onProgressUpdate(values);
        }

        //This block executes in UI when background thread finishes
        //This is where we update the UI with our classification results
        @Override
        protected void onPostExecute(Integer result) {
            super.onPostExecute(result);

            //Hide the progress bar now that we are finished
            ProgressBar bar = (ProgressBar) findViewById(R.id.progressBar);
            bar.setVisibility(View.INVISIBLE);

        }

    }

    private class AsyncTaskLoadModel extends AsyncTaskRunner {

        @Override
        protected void onPreExecute() {
            super.onPreExecute();
            TextView textView = findViewById(R.id.textView);
            String content = TrainModel.id + " is loading model...";
            textView.setText(content);
        }

        @Override
        protected Integer doInBackground(Void... params) {
            try {
//                InputStream is = getResources().openRawResource(R.raw.trained_har_nn);
                TrainModel.model = ModelSerializer.restoreMultiLayerNetwork(
                        TrainModel.locateToLoadModel, false);
            } catch (IOException e) {
                e.printStackTrace();
            }
            return 0;
        }

        @Override
        protected void onPostExecute(Integer result) {
            super.onPostExecute(result);
            TextView textView = findViewById(R.id.textView);
            textView.setText(getString(R.string.model_loaded));
            TrainModel.isTransferred = false;
            isLoading = false;
        }

    }

    private class AsyncTaskTrainModel extends AsyncTaskRunner {

        @Override
        protected void onPreExecute() {
            super.onPreExecute();
            TextView textView = findViewById(R.id.textView);
            String content = TrainModel.id + " is training...";
            textView.setText(content);
        }

        @Override
        protected Integer doInBackground(Void... params) {
            // run training process here
            try {
//                File trainrecords = new File(TrainModel.dir, "WISDM_ar_train_indi_small.csv");
                if (TrainModel.locateToSaveDataSet.length() == 0){
                    // nothing to train
                    return 0;
                }
                MultiLayerNetwork trainedmodel = TrainModel.TrainingModel(TrainModel.locateToSaveDataSet);
                ModelSerializer.writeModel(trainedmodel, new File(TrainModel.syncDir, TrainModel.id + "_updated_model.zip"), false);
                TrainModel.model = trainedmodel;
            } catch (Exception e) {
                e.printStackTrace();
            }
            return 0;
        }

        //This block executes in UI when background thread finishes
        //This is where we update the UI with our classification results
        @Override
        protected void onPostExecute(Integer result) {
            super.onPostExecute(result);

            //Update the UI with output
            TextView textView = findViewById(R.id.textView);
//            textView.setText(String.format("You are %s", LocalDataSet.statues[result]));
            textView.setText(R.string.train_finish);
            isTraining = false;
        }
    }

    private class AsyncTaskLabelling extends  AsyncTaskRunner {
        @Override
        protected void onPreExecute() {
            super.onPreExecute();
            TextView textView = findViewById(R.id.textView);
            String content = TrainModel.id + " is labelling...";
            textView.setText(content);
        }

        @Override
        protected Integer doInBackground(Void... params) {
            try {
                double [] sample = groupSensorData();
                if (sample == null) {
                    return 0;
                }
                int index = LocalDataSet.label2index(currentActivity);
                if (index == -1) {
                    // unexpected label
                    return 0;
                }
                boolean [] indexes = new boolean[6];
                indexes[index] = true;

                try {
                    BufferedWriter br = new BufferedWriter(new FileWriter(
                            TrainModel.locateToSaveDataSet.toString(), true));
                    StringBuilder sb = new StringBuilder();
                    sb.append(String.format("%d", index));
                    for (double element : sample) {
                        sb.append(String.format(",%.4f",element));
                    }
                    br.write(sb.toString());
                    br.newLine();
                    br.close();
                }catch (IOException e) {
                    e.printStackTrace();
                }
//                INDArray sensorData = Nd4j.create(ArrayUtil.flattenDoubleArray(sample), sampleShape);
//                INDArray label = Nd4j.create(indexes);
//                concatTrainingDataSet(sensorData, label);
//                DataSet dataSet = new DataSet(sensorData, label.reshape(labelShape));
//                dataSet.save(new File(TrainModel.dir, "labelled_dataset"));
            } catch (RuntimeException e) {
                e.printStackTrace();
            }
            return 0;
        }

        @Override
        protected void onPostExecute(Integer result) {
            super.onPostExecute(result);
            TextView textView = findViewById(R.id.textView);
            textView.setText(getString(R.string.label_complete));
            isLabelling = false;
        }
    }
}
