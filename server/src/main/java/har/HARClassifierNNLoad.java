package har;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.util.Arrays;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import har.LocalDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

public class HARClassifierNNLoad {

    public static void main(String[] args) throws Exception {

        int batchSize = 50;
        int numOutputs = 6;

        final String filenameTest  = "res/dataset/WISDM_ar_test.csv";

        //Load the test/evaluation data:
        RecordReader rrTest = new CSVRecordReader();
        rrTest.initialize(new FileSplit(new File(filenameTest)));
        DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest,batchSize,0,6);

        File locationToLoad = new File("res/model/trained_har_nn.zip");
//        File locationToLoad = new File("/home/yuze/Desktop/on-device-model/f4dc49973ec1bf39_updated_model.zip");
//        File locationToLoad = new File("/home/yuze/Desktop/on-device-model/7561b62e8b2f8ca9_updated_model.zip");
//        File locationToLoad = new File("res/model/trained_har_nn_updated.zip");
//        File locationToLoad = new File("res/dataset/labelled_dataset.bin");

//        int [] sampleShape = {1, 270};
//        int [] labelShape = {1, 6};
//        boolean [] indexes = new boolean[6];
//        indexes[3] = true;
//        INDArray sensorData = Nd4j.create(ArrayUtil.flattenDoubleArray(LocalDataSet.localdata), sampleShape);
//        INDArray label = Nd4j.create(indexes);
//
//        DataSet dataSet = new DataSet(sensorData, label.reshape(labelShape));

//        DataSet dataSet = new DataSet();
//        dataSet.load(locationToLoad);
//        dataSet.save(locationToLoad);
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(locationToLoad, false);

//        Map<String, INDArray> paramTable = model.paramTable();
//        Set<String> keys = paramTable.keySet();
//        Iterator<String> it = keys.iterator();
//
//        while (it.hasNext()) {
//            String key = it.next();
//            INDArray values = paramTable.get(key);
//            System.out.print(key + " ");//print keys
//            System.out.println(Arrays.toString(values.shape()));//print shape of INDArray
//            System.out.println(values);
//        }

        System.out.println("Evaluate model....");
        Evaluation eval = new Evaluation(numOutputs);
        while(testIter.hasNext()){
            DataSet t = testIter.next();
            INDArray features = t.getFeatures();
            INDArray labels = t.getLabels();
            INDArray predicted = model.output(features,false);

            eval.eval(labels, predicted);

        }

//        Print the evaluation statistics
        System.out.println(eval.stats());

    }
}
