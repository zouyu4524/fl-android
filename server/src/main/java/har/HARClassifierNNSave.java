package har;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;

public class HARClassifierNNSave {

    public static void main(String[] args) throws Exception {
        int seed = 100;
        double learningRate = 0.001;
        int batchSize = 50;
        int nEpochs = 30;

        int numInputs = 270;
        int numOutputs = 6;
        int numHiddenNodes = 1000;

        final String filenameTrain  = "res/dataset/WISDM_ar_train.csv";

        //Load the training data:
        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File(filenameTrain)));
        DataSetIterator trainIter = new RecordReaderDataSetIterator(rr,batchSize,0,6);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .weightInit(WeightInit.XAVIER)
            .updater(new Nesterovs(learningRate, 0.9))
            .list()
            .layer(new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                .activation(Activation.RELU)
                .build())
            .layer(new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                .activation(Activation.RELU)
                .build())
            .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .activation(Activation.SOFTMAX)
                .nIn(numHiddenNodes).nOut(numOutputs).build())
            .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));  //Print score every 10 parameter updates

        model.fit( trainIter, nEpochs );

        boolean saveUpdate = true;

        File locationToSave = new File("res/model/trained_har_nn.zip");

        ModelSerializer.writeModel(model, locationToSave, saveUpdate);

    }
}
