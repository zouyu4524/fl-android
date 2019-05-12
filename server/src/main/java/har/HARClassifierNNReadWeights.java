package har;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Convolution1DLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.util.Map;
import java.util.Set;
import java.util.Iterator;
import java.util.Arrays;

public class HARClassifierNNReadWeights {

    public static void main(String[] args) throws Exception {

        int batchSize = 50;
        int numOutputs = 6;
        int numHiddenNodes = 1000;
        int nEpochs = 30;

        final String filenameTrain  = "res/dataset/WISDM_ar_train.csv";
        final String filenameTest  = "res/dataset/WISDM_ar_test.csv";

        //Load the train data:
        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File(filenameTrain)));
        DataSetIterator trainIter = new RecordReaderDataSetIterator(rr, batchSize, 0, 6);

        //Load the test/evaluation data:
        RecordReader rrTest = new CSVRecordReader();
        rrTest.initialize(new FileSplit(new File(filenameTest)));
        DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest,batchSize,0,6);

        File locationToLoad = new File("res/model/trained_har_nn.zip");

        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(locationToLoad, false);

        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(new Nesterovs(5e-5))
            .seed(100)
            .build();

        MultiLayerNetwork transferred_model = new TransferLearning.Builder(model)
            .fineTuneConfiguration(fineTuneConf)
            .setFeatureExtractor(1)
            .removeOutputLayer()
            .addLayer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .activation(Activation.SOFTMAX)
                .nIn(numHiddenNodes).nOut(numOutputs)
                .weightInit(WeightInit.XAVIER)
                .build())
            .build();

//        transferred_model.init();
        transferred_model.setListeners(new ScoreIterationListener(10));  //Print score every 10 parameter updates
        transferred_model.fit( trainIter, nEpochs );

        Map<String, INDArray> paramTable = transferred_model.paramTable();
        Set<String> keys = paramTable.keySet();
        Iterator<String> it = keys.iterator();

        while (it.hasNext()) {
            String key = it.next();
            INDArray values = paramTable.get(key);
            System.out.print(key + " ");//print keys
            System.out.println(Arrays.toString(values.shape()));//print shape of INDArray
            System.out.println(values);
//            transferred_model.setParam(key, Nd4j.rand(values.shape()));//set some random values
        }

    }
}
