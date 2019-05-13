package com.example.mpchart;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.content.res.Resources;
import android.os.AsyncTask;
import android.os.Environment;
import android.view.View;
import android.widget.ProgressBar;

import androidx.core.app.ActivityCompat;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;

public class TrainModel  {

    public static final File dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS);
    public static final File syncDir = Environment.getExternalStoragePublicDirectory("DropsyncFiles");
    public static final File locateToSaveDataSet = new File(dir, "labelled_dataset.csv");
    public static final File locateToLoadModel = new File(syncDir, "Trained_HAR_NN.zip");
    public static String id = null;

    private static final int numHiddenNodes = 1000;
    private static final int numOutputs = 6;
    private static final int nEpochs = 10;

    public static boolean isTransferred = false;

    public static FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(new Nesterovs(5e-5))
            .seed(100)
            .build();

    public static MultiLayerNetwork model = null;

    public static MultiLayerNetwork TrainingModel(File file) {
        MultiLayerNetwork transferred_model = model;
        if (!isTransferred) {
            transferred_model = new TransferLearning.Builder(model)
                    .fineTuneConfiguration(fineTuneConf)
                    .setFeatureExtractor(1)
                    .build();
            isTransferred = true;
        }
        RecordReader rr = new CSVRecordReader();
        try {
            rr.initialize(new FileSplit(file));
        } catch (Exception e) {
            e.printStackTrace();
        }
        DataSetIterator trainIter = new RecordReaderDataSetIterator(rr, 10, 0, 6);
        transferred_model.fit(trainIter, nEpochs);
        return transferred_model;
    }

}
